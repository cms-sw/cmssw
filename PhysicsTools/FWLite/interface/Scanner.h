// these includes are FWLite-safe
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
// these are from ROOT, so they're safe too
#include <TString.h>
#include <TRegexp.h>
#include <TObjString.h>
#include <TObjArray.h>
#include <TDirectory.h>
#include <TEnv.h>
#include <TClass.h>

#if !defined(__CINT__) && !defined(__MAKECINT__)
#include <RooFit.h>
#include <RooArgList.h>
#include <RooDataSet.h>
#include <RooRealVar.h>
#include <RooCategory.h>

#include "PhysicsTools/FWLite/interface/ScannerHelpers.h"
#endif

#include "PhysicsTools/FWLite/interface/EventSelectors.h"

namespace fwlite {

    /** \brief fwlite::Scanner<C>, a way to inspect or plots elements of a collection C  by using the StringParser. 
     *
     *  fwlite::Scanner<C>, a way to inspect or plots elements of a collection C  by using the StringParser.
     * 
     *  The collection can be something as easy as std::vector<T>, but also some other fancy EDM collections like
     *  RefVector, RefToBaseVector and OwnVector (and probably PtrVector, but it was not tested)
     *
     *  If you're using something other than std::vector, you must provide the full typename, including all 
     *  optional template parameters; e.g. you can't have C = edm::RefVector<reco::MuonCollection>, but you need
     *  C = edm::RefVector<vector<reco::Muon>,reco::Muon,edm::refhelper::FindUsingAdvance<vector<reco::Muon>,reco::Muon> >
     *  In order to figure out what is the correct full name for a collection in an event, open it in ROOT/FWLite,
     *  get the branch name including the trailing ".obj" (hint: Events->GetAlias("label")) usually works),
     *  and then do Events->GetBranch("xxx.obj")->GetClassName() to get something like edm::Wrapper<X>.
     *  then X is what you want to use to create the fwlite::Scanner. Don't use typedefs, they don't work.
     *
     **/
    template<typename Collection>
    class Scanner {
        public:
            /// The type of the Handle to read the Ts from the event. Needed to resolve its Type
            typedef fwlite::Handle<Collection> HandleT;

            /** Create a Scanner, passing a fwlite Event and the labels (just like you would in 'getByLabel') */
            Scanner(fwlite::EventBase *ev, const char *label, const char *instance = "", const char *process="")  :
                event_(ev), label_(label), instance_(instance), 
                printFullEventId_(ev->isRealData()),
                ignoreExceptions_(false),
                exprSep_(":"),
                maxEvents_(-1),
                maxLinesToPrint_(50)
            {
                objType = helper::Parser::elementType(edm::TypeWithDict(HandleT::TempWrapT::typeInfo()));
            }

   //------------------------------------------------------------------------------------------------------------------------------------
            /** Scan the first nmax entries of the event and print out the values of some expressions.
             *
             *  The cut is applied to the individual entries. To set Event-wide cuts, use addEventSelector(). 
             *
             *  The different expressions are separated by ":", unless changed using setExpressionSeparator.
             *  The title of each column is the text of the expression, unless one specifies it differently
             *  by using the notation "@label=expression"
             *
             *  Each row is prefixed by the event id (Run/LS/Event on Data, entry number within the file for MC) 
             *  and by the index of the object within the collection. The behaviour can be changed through the
             *  setPrintFullEventId() method.
             *
             *  The printing will pause by default every 50 lines (see setMaxLinesToPrint() to change this)
             *  Scanning will stop after nmax events.
             */
            void scan(const char *exprs, const char *cut="", int nmax=-1) {
                helper::ScannerBase scanner(objType); 
                scanner.setIgnoreExceptions(ignoreExceptions_);

                TObjArray  *exprArray = TString(exprs).Tokenize(exprSep_);
                int rowline = 0;
                if (printFullEventId_) {
                    printf(" : %9s : %4s : %9s : %3s", "RUN", "LUMI", "EVENT", "#IT");
                    rowline += 3*4+9+4+9+3-1; // -1 as first char remain blank
                } else {
                    printf(" : %5s : %3s", "EVENT", "#IT");
                    rowline += 3+6+3+3-1;  // -1 as first char remain blank
                }
                for (int i = 0; i < exprArray->GetEntries(); ++i) {
                    TString str = ((TObjString *)(*exprArray)[i])->GetString();
                    std::string lb = str.Data();
                    std::string ex = str.Data();
                    if ((ex[0] == '@') && (ex.find('=') != std::string::npos)) {
                        lb = lb.substr(1,ex.find('=')-1); 
                        ex = ex.substr(ex.find('=')+1);    
                    }
                    scanner.addExpression(ex.c_str());
                    printf(" : %8s", (lb.size()>8 ? lb.substr(lb.size()-8) : lb).c_str()); // the rightmost part is usually the more interesting one
                    rowline += 3+8;
                }
                std::cout << " :" << std::endl;
                rowline += 2;
                delete exprArray;

                TString rule('-', rowline);
                std::cout << " " << rule << " " << std::endl;

                if (strlen(cut)) scanner.setCut(cut);

                int iev = 0, line = 0;
                for (event_->toBegin(); (iev != nmax) && !event_->atEnd(); ++iev, ++(*event_)) {
                    if (!selectEvent(*event_)) continue;
                    handle_.getByLabel(*event_, label_.c_str(), instance_.c_str(), process_.c_str());
                    if (handle_.failedToGet()) {
                        if (ignoreExceptions_) continue;
                    } 
                    const Collection & vals = *handle_;
                    for (size_t j = 0, n = vals.size(); j < n; ++j) {
                        if (!scanner.test(&vals[j])) continue;
                        if (printFullEventId_) {
                            const edm::EventAuxiliary &id = event_->eventAuxiliary();
                            printf(" : %9u : %4u : %9llu : %3lu", id.run(), id.luminosityBlock(), id.event(), (unsigned long)j);
                        } else {
			    printf(" : %5d : %3lu", iev, (unsigned long)j);
                        }
                        scanner.print(&vals[j]);
                        std::cout << " :" << std::endl;
                        if (++line == maxLinesToPrint_) {
                            line = 0;
                            if (!wantMore()) { 
                                iev = nmax-1; // this is to exit the outer loop
                                break;        // and this to exit the inner one
                             }
                        }
                    }
                }
                std::cout << std::endl;
            }
  
   //------------------------------------------------------------------------------------------------------------------------------------
            /** Count the number of entries that pass a given cut. 
             *  See setMaxEvents() to specify how many events to loop on when counting.
             *  Events can be further selected by using addEventSelector(). */ 
            size_t count(const char *cut) {
                helper::ScannerBase scanner(objType); 
                scanner.setIgnoreExceptions(ignoreExceptions_);

                scanner.setCut(cut);

                size_t npass = 0;
                int iev = 0;
                for (event_->toBegin(); !event_->atEnd(); ++(*event_), ++iev) {
                    if (maxEvents_ > -1 && iev > maxEvents_) break;
                    if (!selectEvent(*event_)) continue;
                    handle_.getByLabel(*event_, label_.c_str(), instance_.c_str(), process_.c_str());
                    const Collection & vals = *handle_;
                    for (size_t j = 0, n = vals.size(); j < n; ++j) {
                        if (scanner.test(&vals[j])) npass++;
                    }
                }
                return npass;
            }

            /** Count the number of events, taking into account setMaxEvents() and the event selectors */
            size_t countEvents() {
                size_t npass = 0;
                int iev = 0;
                for (event_->toBegin(); !event_->atEnd(); ++(*event_), ++iev) {
                    if (maxEvents_ > -1 && iev > maxEvents_) break;
                    if (selectEvent(*event_)) npass++;
                }
                return npass;
            }

   //------------------------------------------------------------------------------------------------------------------------------------
            /** Plot the expression expr for events passing 'cut, into histogram hist.
             *  hist is *not* reset before filling it, so it will add to the existing content.
             *
             *  If "NORM" is specified in the draw options, the output histogram is normalized
             *  If "GOFF" is specified in the draw options, the output histogram is not drawn
             *
             *  See setMaxEvents() to specify how many events to loop on when plotting.
             *  Events can be further selected by using addEventSelector(). */ 
            TH1 * draw(const char *expr, const char *cut, TString drawopt, TH1 *hist) {
                // prep the machinery
                helper::ScannerBase scanner(objType);
                scanner.setIgnoreExceptions(ignoreExceptions_);
                if (!scanner.addExpression(expr)) return 0;
                if (strlen(cut)) scanner.setCut(cut);

                // check histo
                if (hist == 0) {
                    std::cerr << "Method draw(expr, cut, drawopt, hist) cannot be called with null 'hist'. Use the other draw methods instead." << std::endl;
                    return 0;
                }

                // fill histogram
                int iev = 0;
                for (event_->toBegin(); !event_->atEnd(); ++(*event_), ++iev) {
                    if (maxEvents_ > -1 && iev > maxEvents_) break;
                    if (!selectEvent(*event_)) continue;
                    handle_.getByLabel(*event_, label_.c_str(), instance_.c_str(), process_.c_str());
                    const Collection & vals = *handle_;
                    for (size_t j = 0, n = vals.size(); j < n; ++j) {
                        scanner.fill1D(&vals[j], hist);
                    }
                }

                if (drawopt.Contains("NORM",TString::kIgnoreCase) && (hist->Integral() != 0)) {
                    hist->Sumw2();
                    hist->Scale(1.0/hist->Integral());
                    // remove the "NORM" because THistPainter doesn't understand it
                    drawopt(TRegexp("[Nn][Oo][Rr][Mm]")) = "";
                }

                if (!drawopt.Contains("GOFF",TString::kIgnoreCase)) hist->Draw(drawopt);
                return hist;
            }

            /** Plot the expression 'expr' for events passing 'cut', in a histogram named 'hname'
             *  - If htemplate is provided it will be cloned, 
             *  - otherwise, if "SAME" is among the draw options, it will clone "htemp" (if it's available)
             *  - otherwise an automatically binned histogram will be used.
             *    in the last case, gEnv->GetValue("Hist.Binning.1D.x", 100) is used for the number of bins
             *  See draw(const char *expr, const char *cut, TString drawopt, TH1 *histo) for further documentation */
            TH1 * draw(const char *expr, const char *cut = "", TString drawopt = "", const char *hname = "htemp", const TH1 *htemplate = 0) {
                TH1 *hist = 0;
                if (htemplate != 0) {
                    if ((strcmp(hname, "htemp") == 0) && (strcmp(hname,htemplate->GetName()) != 0)) htempDelete();
                    hist = (TH1*) hist->Clone(hname);
                } else if (drawopt.Contains("SAME",TString::kIgnoreCase)) { 
                    hist = getSameH1(hname);
                }

                // if in the end we found no way to make "hist"
                if (hist == 0) {
                    if (strcmp(hname, "htemp") == 0) htempDelete();
                    hist = new TH1F(hname, "", gEnv->GetValue("Hist.Binning.1D.x",100), 0, 0);
                    hist->SetCanExtend(TH1::kAllAxes);
                }
                hist->SetTitle((strlen(cut) ? TString(expr)+"{"+cut+"}" : TString(expr)));
                hist->GetXaxis()->SetTitle(expr);
                return draw(expr, cut, drawopt, hist);
            }


            /// Make a histogram named hname with nbins from xlow to xhigh, and then call draw().
            /// If "SAME" is passed in the draw options, complain and ignore the binning.
            TH1 * draw(const char *expr, int nbins, double xlow, double xhigh, const char *cut = "", const char *drawopt = "", const char *hname = "htemp") {
                if (TString(drawopt).Contains("SAME",TString::kIgnoreCase)) { 
                    std::cerr << "Binning is ignored when 'SAME' is specified." << std::endl; 
                    TH1 *hsame = getSameH1(hname);
                    return draw(expr, cut, drawopt, hsame);
                }
                if (strcmp(hname, "htemp") == 0) htempDelete();
                TH1 * htemp = new TH1F(hname, expr, nbins, xlow, xhigh);
                if (strlen(cut)) htemp->SetTitle(TString(expr)+"{"+cut+"}");
                htemp->GetXaxis()->SetTitle(expr);
                return draw(expr,cut,drawopt,htemp);
            }
            /// Make a histogram named hname with nbins with boundaries xbins, and then call draw().
            /// If "SAME" is passed in the draw options, complain and ignore the binning.
            TH1 * draw(const char *expr, int nbins, double *xbins, const char *cut = "", const char *drawopt = "", const char *hname = "htemp") {
                if (TString(drawopt).Contains("SAME",TString::kIgnoreCase)) { 
                    std::cerr << "Binning is ignored when 'SAME' is specified." << std::endl; 
                    TH1 *hsame = getSameH1(hname);
                    return draw(expr, cut, drawopt, hsame);
                }
                if (strcmp(hname, "htemp") == 0) htempDelete();
                TH1 * htemp = new TH1F(hname, expr, nbins, xbins);
                if (strlen(cut)) htemp->SetTitle(TString(expr)+"{"+cut+"}");
                htemp->GetXaxis()->SetTitle(expr);
                return draw(expr,cut,drawopt,htemp);
            }

   //------------------------------------------------------------------------------------------------------------------------------------
            /// Just like draw() except that it uses TProfile. Note that the order is (x,y) while in ROOT it's usually (y,x)!
            TProfile * drawProf(TString xexpr, TString yexpr, const char *cut, TString drawopt, TProfile *hist) {
                // prep the machinery
                helper::ScannerBase scanner(objType);
                scanner.setIgnoreExceptions(ignoreExceptions_);
                if (!scanner.addExpression(xexpr.Data())) return 0;
                if (!scanner.addExpression(yexpr.Data())) return 0;
                if (strlen(cut)) scanner.setCut(cut);

                // check histo
                if (hist == 0) {
                    std::cerr << "Method drawProf(xexpr, yexpr, cut, drawopt, hist) cannot be called with null 'hist'. Use the other draw methods instead." << std::endl;
                    return 0;
                }

                // fill histogram
                int iev = 0;
                for (event_->toBegin(); !event_->atEnd(); ++(*event_), ++iev) {
                    if (maxEvents_ > -1 && iev > maxEvents_) break;
                    if (!selectEvent(*event_)) continue;
                    handle_.getByLabel(*event_, label_.c_str(), instance_.c_str(), process_.c_str());
                    const Collection & vals = *handle_;
                    for (size_t j = 0, n = vals.size(); j < n; ++j) {
                        scanner.fillProf(&vals[j], hist);
                    }
                }

                if (!strlen(hist->GetTitle())) hist->SetTitle((strlen(cut) ? yexpr+":"+xexpr+"{"+cut+"}" : yexpr+":"+xexpr));
                if (!strlen(hist->GetXaxis()->GetTitle())) hist->GetXaxis()->SetTitle(xexpr);
                if (!strlen(hist->GetYaxis()->GetTitle())) hist->GetYaxis()->SetTitle(yexpr);
                if (!TString(drawopt).Contains("GOFF",TString::kIgnoreCase)) hist->Draw(drawopt);
                return hist;
            }
            /// Just like draw() except that it uses TProfile. Note that the order is (x,y) while in ROOT it's usually (y,x)!
            TProfile * drawProf(TString xexpr, TString yexpr, const char *cut = "", TString drawopt = "", const char *hname = "htemp", TProfile *htemplate = 0) {
                TProfile *hist = 0;
                if (htemplate != 0) {
                    if ((strcmp(hname, "htemp") == 0) && (strcmp(hname,htemplate->GetName() )!= 0)) htempDelete();
                    hist = (TProfile*) hist->Clone(hname);
                } else if (drawopt.Contains("SAME",TString::kIgnoreCase)) { 
                    hist = getSameProf(hname);
                }

                // if in the end we found no way to make "hist"
                if (hist == 0) {
                    if (strcmp(hname, "htemp") == 0) htempDelete();
                    hist = new TProfile(hname, "", gEnv->GetValue("Hist.Binning.1D.x",100), 0., 0.);
                    hist->SetCanExtend(TH1::kAllAxes);
                }
                return drawProf(xexpr, yexpr, cut, drawopt, hist);
            }

            /// Just like draw() except that it uses TProfile. Note that the order is (x,y) while in ROOT it's usually (y,x)!
            TProfile * drawProf(TString xexpr, int bins, double xlow, double xhigh, TString yexpr, const char *cut = "", const char *drawopt = "", const char *hname = "htemp") {
                if (TString(drawopt).Contains("SAME",TString::kIgnoreCase)) { 
                    std::cerr << "Binning is ignored when 'SAME' is specified." << std::endl; 
                    TProfile *hsame = getSameProf(hname);
                    return drawProf(xexpr, yexpr, cut, drawopt, hsame);
                }
                if (strcmp(hname, "htemp") == 0) htempDelete();
                TProfile * htemp = new TProfile(hname, "", bins, xlow, xhigh);
                return drawProf(xexpr,yexpr,cut,drawopt,htemp);
            }

   //------------------------------------------------------------------------------------------------------------------------------------
            /// Just like draw() except that it uses TH2. Note that the order is (x,y) while in ROOT it's usually (y,x)!
            TH2 * draw2D(TString xexpr, TString yexpr, const char *cut, TString drawopt, TH2 *hist) {
                // prep the machinery
                helper::ScannerBase scanner(objType);
                scanner.setIgnoreExceptions(ignoreExceptions_);
                if (!scanner.addExpression((const char *)xexpr)) return 0;
                if (!scanner.addExpression((const char *)yexpr)) return 0;
                if (strlen(cut)) scanner.setCut(cut);

                // check histo
                if (hist == 0) {
                    std::cerr << "Method draw2D(xexpr, yexpr, cut, drawopt, hist) cannot be called with null 'hist'. Use the other draw methods instead." << std::endl;
                    return 0;
                }

                // fill histogram
                int iev = 0;
                for (event_->toBegin(), iev = 0; !event_->atEnd(); ++(*event_), ++iev) {
                    if (maxEvents_ > -1 && iev > maxEvents_) break;
                    if (!selectEvent(*event_)) continue;
                    handle_.getByLabel(*event_, label_.c_str(), instance_.c_str(), process_.c_str());
                    const Collection & vals = *handle_;
                    for (size_t j = 0, n = vals.size(); j < n; ++j) {
                        scanner.fill2D(&vals[j], hist);
                    }
                }

                if (!strlen(hist->GetTitle())) hist->SetTitle((strlen(cut) ? yexpr+":"+xexpr+"{"+cut+"}" : yexpr+":"+xexpr));
                if (!strlen(hist->GetXaxis()->GetTitle())) hist->GetXaxis()->SetTitle(xexpr);
                if (!strlen(hist->GetYaxis()->GetTitle())) hist->GetYaxis()->SetTitle(yexpr);
                if (!TString(drawopt).Contains("GOFF",TString::kIgnoreCase)) hist->Draw(drawopt);
                return hist;
            }
            /// Just like draw() except that it uses TH2. Note that the order is (x,y) while in ROOT it's usually (y,x)!
            /// Note that automatical binning for TH2s is more expensive, as it requires to loop on the events twice!
            TH2 * draw2D(TString xexpr, TString yexpr, const char *cut = "", TString drawopt = "", const char *hname = "htemp", TH2 *htemplate = 0) {
                TH2 *hist = 0;
                if (htemplate != 0) {
                    if ((strcmp(hname, "htemp") == 0) && (strcmp(hname,htemplate->GetName()) != 0)) htempDelete();
                    hist = (TH2*) hist->Clone(hname);
                } else if (drawopt.Contains("SAME",TString::kIgnoreCase)) { 
                    hist = getSameH2(hname);
                }

                // if in the end we found no way to make "hist"
                if (hist == 0) {
                    // prep the machinery
                    helper::ScannerBase scanner(objType);
                    scanner.setIgnoreExceptions(ignoreExceptions_);
                    if (!scanner.addExpression((const char *)xexpr)) return 0;
                    if (!scanner.addExpression((const char *)yexpr)) return 0;
                    if (strlen(cut)) scanner.setCut(cut);

                    if (strcmp(hname, "htemp") == 0) htempDelete();
                    // ok this is much more a hack than for the 1D case
                    double xmin = 0, xmax = -1, ymin = 0, ymax = -1; int iev;
                    for (event_->toBegin(), iev = 0; !event_->atEnd(); ++(*event_), ++iev) {
                        if (maxEvents_ > -1 && iev > maxEvents_) break;
                        if (!selectEvent(*event_)) continue;
                        handle_.getByLabel(*event_, label_.c_str(), instance_.c_str(), process_.c_str());
                        const Collection & vals = *handle_;
                        for (size_t j = 0, n = vals.size(); j < n; ++j) {
                            if (!scanner.test(&vals[j])) continue;
                            double x = scanner.eval(&vals[j],0);
                            double y = scanner.eval(&vals[j],1);
                            if ((xmax == -1) || (x >= xmax)) xmax = x;
                            if ((xmin ==  0) || (x <= xmin)) xmin = x;
                            if ((ymax == -1) || (y >= ymax)) ymax = y;
                            if ((ymin ==  0) || (y <= ymin)) ymin = y;
                        }
                    }
                    hist = new TH2F(hname, "",
                            gEnv->GetValue("Hist.Binning.2D.x",20), xmin, xmax,
                            gEnv->GetValue("Hist.Binning.2D.y",20), ymin, ymax);
                }
                return draw2D(xexpr, yexpr, cut, drawopt, hist);
            }

            /// Just like draw() except that it uses TH2. Note that the order is (x,y) while in ROOT it's usually (y,x)!
            TH2 * draw2D(TString xexpr, int xbins, double xlow, double xhigh, 
                         TString yexpr, int ybins, double ylow, double yhigh,
                         const char *cut = "", const char *drawopt = "", const char *hname="htemp") {
                if (TString(drawopt).Contains("SAME",TString::kIgnoreCase)) { 
                    std::cerr << "Binning is ignored when 'SAME' is specified." << std::endl; 
                    TH2 *hsame = getSameH2(hname);
                    return draw2D(xexpr, yexpr, cut, drawopt, hsame);
                }
                if (strcmp(hname, "htemp") == 0) htempDelete();
                TH2 * htemp = new TH2F("htemp", "", xbins, xlow, xhigh, ybins,ylow,yhigh);
                return draw2D(xexpr,yexpr,cut,drawopt,htemp);
            }

            /** Draw a scatter plot of x vs y for events passing the cut. */
            TGraph * drawGraph(TString xexpr, TString yexpr, const char *cut, TString drawopt, TGraph *graph) {
                // prep the machinery
                helper::ScannerBase scanner(objType);
                scanner.setIgnoreExceptions(ignoreExceptions_);
                if (!scanner.addExpression((const char *)xexpr)) return 0;
                if (!scanner.addExpression((const char *)yexpr)) return 0;
                if (strlen(cut)) scanner.setCut(cut);

                // make graph, if needed
                if (graph == 0) {
                    graph = new TGraph();
                    graph->SetNameTitle("htemp", (strlen(cut) ? yexpr+":"+xexpr+"{"+cut+"}" : yexpr+":"+xexpr)); 
                }

                // fill graph
                int iev = 0;
                for (event_->toBegin(); !event_->atEnd(); ++(*event_), ++iev) {
                    if (maxEvents_ > -1 && iev > maxEvents_) break;
                    if (!selectEvent(*event_)) continue;
                    handle_.getByLabel(*event_, label_.c_str(), instance_.c_str(), process_.c_str());
                    const Collection & vals = *handle_;
                    for (size_t j = 0, n = vals.size(); j < n; ++j) {
                        scanner.fillGraph(&vals[j], graph);
                    }
                }

                if (!strlen(graph->GetTitle())) graph->SetTitle((strlen(cut) ? yexpr+":"+xexpr+"{"+cut+"}" : yexpr+":"+xexpr));
                if (!strlen(graph->GetXaxis()->GetTitle())) graph->GetXaxis()->SetTitle(xexpr);
                if (!strlen(graph->GetYaxis()->GetTitle())) graph->GetYaxis()->SetTitle(yexpr);
                if (!TString(drawopt).Contains("GOFF",TString::kIgnoreCase)) graph->Draw(drawopt);
                return graph;
            }

            /** Draw a scatter plot of x vs y for events passing the cut. */
            TGraph * drawGraph(TString xexpr, TString yexpr, const char *cut = "", TString drawopt = "AP", const char *gname = "htemp") {
                if (strcmp(gname, "htemp") == 0) htempDelete();
                TGraph *graph =  new TGraph();
                graph->SetNameTitle(gname, (strlen(cut) ? yexpr+":"+xexpr+"{"+cut+"}" : yexpr+":"+xexpr)); 
                return drawGraph(xexpr,yexpr,cut,drawopt,graph);
            }


   //------------------------------------------------------------------------------------------------------------------------------------
            /** Fill a RooDataSet.
             *  - Real variables are defined just like in the scan() command; a list separated by ":" (see also setExpressionSeparator()); 
             *  - Boolean variables are defined just like cuts, and are created as RooCategory with two states: pass(1) and fail(0).
             *  For each variable, the name is taken from the expression itself, or can be specified manuall by using the notation "@name=expr"
             *  Note: the dataset contains one entry per item, irrespectively of how entries are distributed among events.
             */
            RooDataSet *fillDataSet(const char *realvars, const char *boolvars, const char *cut="", const char *name="data") {
                helper::ScannerBase scanner(objType); 
                scanner.setIgnoreExceptions(ignoreExceptions_);

                RooArgList vars;
                TObjArray  *exprArray = TString(realvars).Tokenize(exprSep_);
                TObjArray  *catArray  = TString(boolvars).Tokenize(exprSep_);
                int nreals = exprArray->GetEntries();
                int nbools  = catArray->GetEntries();
                for (int i = 0; i < nreals; ++i) {
                    TString str = ((TObjString *)(*exprArray)[i])->GetString();
                    std::string lb = str.Data();
                    std::string ex = str.Data();
                    if ((ex[0] == '@') && (ex.find('=') != std::string::npos)) {
                        lb = lb.substr(1,ex.find('=')-1); 
                        ex = ex.substr(ex.find('=')+1);    
                    }
                    if (!scanner.addExpression(ex.c_str())) {
                        std::cerr << "Filed to define real variable '" << lb << "', expr = '" << ex << "'" << std::endl;
                        return 0;
                    }
                    // FIXME: I have to leave it dangling on the HEAP otherwise ROOT segfaults...
                    RooRealVar *var = new RooRealVar(lb.c_str(),lb.c_str(), 0.0);
                    vars.add(*var);
                }
                for (int i = 0; i < nbools; ++i) {
                    TString str = ((TObjString *)(*catArray)[i])->GetString();
                    std::string lb = str.Data();
                    std::string ex = str.Data();
                    if ((ex[0] == '@') && (ex.find('=') != std::string::npos)) {
                        lb = lb.substr(1,ex.find('=')-1); 
                        ex = ex.substr(ex.find('=')+1);    
                    }
                    if (!scanner.addExtraCut(ex.c_str())) {
                        std::cerr << "Filed to define bool variable '" << lb << "', cut = '" << ex << "'" << std::endl;
                        return 0;
                    }
                    RooCategory *cat = new RooCategory(lb.c_str(), lb.c_str());
                    cat->defineType("fail",0);
                    cat->defineType("pass",1);
                    vars.add(*cat);
                }

                RooDataSet *ds = new RooDataSet(name, name, vars);

                if (strlen(cut)) scanner.setCut(cut);
                int iev = 0;
                for (event_->toBegin(); !event_->atEnd(); ++iev, ++(*event_)) {
                    if (maxEvents_ > -1 && iev > maxEvents_) break;
                    if (!selectEvent(*event_)) continue;
                    handle_.getByLabel(*event_, label_.c_str(), instance_.c_str(), process_.c_str());
                    if (handle_.failedToGet()) {
                        if (ignoreExceptions_) continue;
                    } 
                    const Collection & vals = *handle_;
                    for (size_t j = 0, n = vals.size(); j < n; ++j) {
                        if (!scanner.test(&vals[j])) continue;
                        for (int i = 0; i < nreals; ++i) {
                            RooRealVar *var = (RooRealVar *)vars.at(i);
                            var->setVal(scanner.eval(&vals[j], i));
                        }
                        for (int i = 0; i < nbools; ++i) {
                            RooCategory *cat = (RooCategory*) vars.at(i+nreals);
                            cat->setIndex(int(scanner.test(&vals[j], i+1))); // 0 is the event selection cut
                        }
                        ds->add(vars);
                    }
                }
    
                delete exprArray;
                delete catArray;

                return ds;
            }
  


            void setPrintFullEventId(bool printIt=true) { printFullEventId_ = printIt; }
            void setExpressionSeparator(TString separator) { exprSep_ = separator; }
            void setIgnoreExceptions(bool ignoreThem) { ignoreExceptions_ = ignoreThem; }
            void setMaxLinesToPrint(int lines) { maxLinesToPrint_ = (lines > 0 ? lines : 2147483647); }
    
            void addEventSelector(fwlite::EventSelector *selector) { eventSelectors_.Add(selector); }
            void clearEventSelector() { eventSelectors_.Clear(); }
            TObjArray & eventSelectors() { return eventSelectors_; }
            bool selectEvent(const fwlite::EventBase &ev) const {
                for (int i = 0, n = eventSelectors_.GetEntries(); i < n; ++i) {
                    if (!((fwlite::EventSelector *)(eventSelectors_[i]))->accept(ev)) return false;
                }
                return true;
            }

            void setMaxEvents(int max) { maxEvents_ = max; }
        private:
            fwlite::EventBase *event_;
            std::string    label_, instance_, process_;
            bool printFullEventId_;
            bool ignoreExceptions_;
            TString exprSep_;
            HandleT        handle_;
            edm::TypeWithDict   objType;

            TObjArray eventSelectors_;

            int maxEvents_;

            int maxLinesToPrint_;
            bool wantMore() const {
                // ask if user wants more
                fprintf(stderr,"Type <CR> to continue or q to quit ==> ");
                // read first char
                int readch = getchar(), answer = readch;
                // poll out remaining chars from buffer
                while (readch != '\n' && readch != EOF) readch = getchar();
                // check first char
                return !(answer == 'q' || answer == 'Q');
            }

            void htempDelete() {
                if (gDirectory) {
                    TObject *obj = gDirectory->Get("htemp");
                    if (obj) obj->Delete();
                }
            }

            /// Get whatever histogram makes sense for a plot passing "SAME" in drawOpt, and call it hname
            /// Currently it won't work if the histogram of which we want to be "SAME" is not called "htemp"
            TH1 *getSameH1(const char *hname) {
                if (gDirectory && gDirectory->Get("htemp") != 0 && 
                        gDirectory->Get("htemp")->IsA()->InheritsFrom(TH1::Class())) {
                    TH1 *hist = (TH1*) ((TH1*) gDirectory->Get("htemp"))->Clone(hname);
                    hist->Reset();
                    hist->SetLineColor(kBlack);
                    hist->SetMarkerColor(kBlack);
                    return hist;
                } else {
                    std::cerr << "There is no 'htemp' histogram from which to 'SAME'." << std::endl;
                    return 0;
                }
            }

            /// Get whatever histogram makes sense for a plot passing "SAME" in drawOpt, and call it hname
            /// Currently it won't work if the histogram of which we want to be "SAME" is not called "htemp"
            TH2 *getSameH2(const char *hname) {
                if (gDirectory && gDirectory->Get("htemp") != 0 && 
                        gDirectory->Get("htemp")->IsA()->InheritsFrom(TH2::Class())) {
                    TH2 *hist = (TH2*) ((TH2*) gDirectory->Get("htemp"))->Clone(hname);
                    hist->Reset();
                    hist->SetLineColor(kBlack);
                    hist->SetMarkerColor(kBlack);
                    return hist;
                } else {
                    std::cerr << "There is no 'htemp' histogram from which to 'SAME'." << std::endl;
                    return 0;
                }
            }

            /// Get whatever histogram makes sense for a plot passing "SAME" in drawOpt, and call it hname
            /// Currently it won't work if the histogram of which we want to be "SAME" is not called "htemp"
            TProfile *getSameProf(const char *hname) {
                if (gDirectory && gDirectory->Get("htemp") != 0 && 
                        gDirectory->Get("htemp")->IsA()->InheritsFrom(TProfile::Class())) {
                    TProfile *hist = (TProfile*) ((TProfile*) gDirectory->Get("htemp"))->Clone(hname);
                    hist->Reset();
                    hist->SetLineColor(kBlack);
                    hist->SetMarkerColor(kBlack);
                    return hist;
                } else {
                    std::cerr << "There is no 'htemp' histogram from which to 'SAME'." << std::endl;
                    return 0;
                }
            }


    };
}
