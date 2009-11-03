#include "CalibTracker/SiStripLorentzAngle/interface/LA_Filler_Fitter.h"
#include "CalibTracker/SiStripCommon/interface/TTREE_FOREACH_ENTRY.hh"
#include "CalibTracker/SiStripCommon/interface/Book.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <cmath>
#include <boost/foreach.hpp>
#include <boost/regex.hpp> 
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/erase.hpp>
#include <TF1.h>
#include <TTree.h>
#include <TGraphErrors.h>
#include <TProfile.h>

void LA_Filler_Fitter::
fill(TTree* tree, Book& book) {
  TTREE_FOREACH_ENTRY(tree,
		      const std::vector<unsigned>* LEAF( clusterdetid )
		      const std::vector<unsigned>* LEAF( clusterwidth )
		      const std::vector<float>*    LEAF( clustervariance )
		      const std::vector<unsigned>* LEAF( tsostrackmulti )
		      const std::vector<float>*    LEAF( tsosdriftx )
		      const std::vector<float>*    LEAF( tsosdriftz )
		      const std::vector<float>*    LEAF( tsoslocaltheta )
		      const std::vector<float>*    LEAF( tsoslocalphi )
		      const std::vector<float>*    LEAF( tsosBdotY )
		      const std::vector<float>*    LEAF( tsosglobalZofunitlocalY )
		      ) {
    if(maxEvents_) TTREE_FOREACH_ENTRY_total = std::min(maxEvents_,TTREE_FOREACH_ENTRY_total);
    for(unsigned i=0; i< clusterwidth->size() ; i++) {

      const SiStripDetId detid((*clusterdetid)[i]);

      if( (*tsostrackmulti)[i] != 1 ||
	  (detid.subDetector()!=SiStripDetId::TIB && 
	   detid.subDetector()!=SiStripDetId::TOB)    ) 
	continue;

      const double tthetaL = (*tsosdriftx)[i] / (*tsosdriftz)[i];
      const double tthetaT = tan((*tsoslocaltheta)[i]) * cos((*tsoslocalphi)[i]);

      const int sign = (*tsosglobalZofunitlocalY)[i] < 0  ?  -1  :  1 ;
      const unsigned width = (*clusterwidth)[i];
      const double var = (*clustervariance)[i];

      poly<std::string> granular;
      granular += subdetLabel(detid);
      if(byLayer_)  granular *= layerLabel(detid);
      if(byModule_) granular *= moduleLabel(detid);
      if(ensembleBins_) {
	granular+= "_ensembleBin"+boost::lexical_cast<std::string>((int)(ensembleBins_*(sign*tthetaL-ensembleLow_)/(ensembleUp_-ensembleLow_)));
	granular+= "";
	if(ensembleSize_) granular*= "_sample"+boost::lexical_cast<std::string>(TTREE_FOREACH_ENTRY_index % ensembleSize_);
      }

      poly<std::string> VW; VW++;
      poly<std::string> A1("_all");  if(width==1 && methods_ & PROB1    )      A1*="_w1";
                                else if(width==2 && methods_ & (AVGV2|RMSV2) ) VW*=method(AVGV2,0);
				else if(width==3 && methods_ & (AVGV3|RMSV3) ) VW*=method(AVGV3,0);

                            book.fill( sign*tthetaL,                    granular+"_reconstruction" ,360,-1.0,1.0 );
                            book.fill( sign*(tthetaT-tthetaL),          granular+A1                ,360,-1.0,1.0 );
                            book.fill( sign*(tthetaT-tthetaL),     var, granular+VW                ,360,-1.0,1.0 );
      if(methods_ & WIDTH)  book.fill( sign*tthetaT,             width, granular+method(WIDTH)     , 81,-0.6,0.6 );
      if(ensembleBins_==0)  book.fill( fabs((*tsosBdotY)[i]),           granular+"_field"          ,101,   1,  5 );
    }
  }
}

std::string LA_Filler_Fitter::
subdetLabel(const SiStripDetId detid) { return detid.subDetector()==SiStripDetId::TOB? "TOB" : "TIB";}
std::string LA_Filler_Fitter::
moduleLabel(const SiStripDetId detid) { return subdetLabel(detid) + "_module"+boost::lexical_cast<std::string>(detid());}
std::string LA_Filler_Fitter::
layerLabel(const SiStripDetId detid) {
  unsigned layer = detid.subDetector() == SiStripDetId::TOB ? TOBDetId(detid()).layer() : TIBDetId(detid()).layer();
  return subdetLabel(detid)+"_layer"+boost::lexical_cast<std::string>(layer)+(detid.stereo()?"s":"a");
}

void LA_Filler_Fitter::
fit_width_profile(Book& book) {
  for(Book::iterator it = book.begin(".*"+method(WIDTH)); it!=book.end(); ++it) {
    TH1* const p = it->second;
    if(p->GetEntries() < 400) { delete p; book[it->first]=0; continue;}
    p->SetTitle(";tan#theta_{t};");
    const float min = p->GetMinimum();
    const float max = p->GetMaximum();
    float xofmin = p->GetBinCenter(p->GetMinimumBin()); if( xofmin>0.0 || xofmin<-0.15) xofmin = -0.05;
    const float xofmax = p->GetBinCenter(p->GetMaximumBin());

    TF1* const fit = new TF1("LA_profile_fit","[2]*(TMath::Abs(x-[0]))+[1]",-1,1);
    fit->SetParLimits(0,-0.15,0.01);
    fit->SetParLimits(1, 0.6*min, 1.25*min );
    fit->SetParLimits(2,0.1,10);
    fit->SetParameters( xofmin, min, (max-min) / fabs( xofmax - xofmin ) );

    int badfit = p->Fit(fit,"IEQ","",-.5,.3);
    if( badfit ) badfit = p->Fit(fit,"IEQ","", -.46,.26);
    if( badfit ) {delete p; book[it->first]=0;}
  }
}

void LA_Filler_Fitter::
make_and_fit_symmchi2(Book& book) {
  for(Book::iterator it = book.begin(".*_all"); it!=book.end(); ++it) {
    const std::string base = boost::erase_all_copy(it->first,"_all");

    std::vector<Book::iterator> rebin_hists;              
    Book::iterator    all = it;	                           rebin_hists.push_back(all);   
    Book::iterator     w1 = book.find(base+"_w1");         rebin_hists.push_back(w1);    
    Book::iterator var_w2 = book.find(base+method(AVGV2,0)); rebin_hists.push_back(var_w2);
    Book::iterator var_w3 = book.find(base+method(AVGV3,0)); rebin_hists.push_back(var_w3);

    const unsigned rebin = std::max( var_w2==book.end() ? 0 : find_rebin(var_w2->second), 
				     var_w3==book.end() ? 0 : find_rebin(var_w3->second) );
    BOOST_FOREACH(Book::iterator it, rebin_hists) if(it!=book.end()) it->second->Rebin( rebin>1 ? rebin<7 ? rebin : 6 : 1);

    TH1* const prob_w1 = w1==book.end()     ? 0 : subset_probability( base+method(PROB1,0) ,w1->second,all->second);         book.book(base+method(PROB1,0),prob_w1);
    TH1* const rmsv_w2 = var_w2==book.end() ? 0 :        rms_profile( base+method(RMSV2,0), (TProfile*const)var_w2->second); book.book(base+method(RMSV2,0),rmsv_w2);
    TH1* const rmsv_w3 = var_w3==book.end() ? 0 :        rms_profile( base+method(RMSV3,0), (TProfile*const)var_w3->second); book.book(base+method(RMSV3,0),rmsv_w3);
    
    //book.erase(all);
    //if(w1!=book.end()) book.erase(w1);

    std::vector<TH1*> fit_hists;
    if(prob_w1) {
      fit_hists.push_back(prob_w1);  prob_w1->SetTitle("Width==1 Probability;tan#theta_{t}-(dx/dz)_{reco}");
    }
    if(var_w2!=book.end())  {
      fit_hists.push_back(var_w2->second);   var_w2->second->SetTitle("Width==2 Mean Variance;tan#theta_{t}-(dx/dz)_{reco}");
      fit_hists.push_back(rmsv_w2);                 rmsv_w2->SetTitle("Width==2 RMS Variance;tan#theta_{t}-(dx/dz)_{reco}");
    }
    if(var_w3!=book.end())  {
      fit_hists.push_back(var_w3->second);   var_w3->second->SetTitle("Width==3 Mean Variance;tan#theta_{t}-(dx/dz)_{reco}");
      fit_hists.push_back(rmsv_w3);                 rmsv_w3->SetTitle("Width==3 RMS Variance;tan#theta_{t}-(dx/dz)_{reco}");
    }

    const unsigned bins = fit_hists[0]->GetNbinsX();
    const unsigned guess = fit_hists[0]->FindBin(0);
    const std::pair<unsigned,unsigned> range(guess-bins/30,guess+bins/30-1);

    BOOST_FOREACH(TH1*const hist, fit_hists) {
      TH1*const chi2 = SymmetryFit::symmetryChi2(hist,range);
      if(chi2) {book.book(chi2->GetName(),chi2); chi2->SetTitle("Symmetry #chi^{2};tan#theta_{t}-(dx/dz)_{reco}");}
    }
    //TH1* const chi2 = SymmetryFit::symmetryChi2(base+method(MULTI,0), fit_hists, range);
    //if(chi2) { book.book(SymmetryFit::name(base+method(MULTI,0)), chi2); chi2->SetTitle("MultiSymmetry #chi^{2};tan#theta_{t}-(dx/dz)_{reco}");}
  }
}

unsigned LA_Filler_Fitter::
find_rebin(const TH1* const hist) {
  const double mean = hist->GetMean();
  const double rms = hist->GetRMS();
  const int begin = std::min(                1, hist->GetXaxis()->FindFixBin(mean-rms));
  const int end   = std::max(hist->GetNbinsX(), hist->GetXaxis()->FindFixBin(mean+rms)) + 1;
  unsigned current_hole(0), max_hole(0);
  for(int i=begin; i<end; i++) {
    if(!hist->GetBinError(i)) current_hole++;
    else if(current_hole) {max_hole = std::max(current_hole,max_hole); current_hole=0;}
  }
  return max_hole+1;
}

TH1* LA_Filler_Fitter::
rms_profile(const std::string name, const TProfile* const prof) {
  const int bins = prof->GetNbinsX();
  TH1* const rms = new TH1F(name.c_str(),"",bins, prof->GetBinLowEdge(1),  prof->GetBinLowEdge(bins) + prof->GetBinWidth(bins) );
  for(int i = 1; i<=bins; i++) {
    const double Me = prof->GetBinError(i);
    const double neff = prof->GetBinEntries(i); //Should be prof->GetBinEffectiveEntries(i);, not availible this version ROOT.  This is only ok for unweighted fills
    rms->SetBinContent(i, Me*sqrt(neff) );
    rms->SetBinError(i, Me/sqrt(2) );
  }
  return rms;
}

TH1* LA_Filler_Fitter::
subset_probability(const std::string name, const TH1* const subset, const TH1* const total) {
  const int bins = subset->GetNbinsX();
  TH1* const prob = new TH1F(name.c_str(),"",bins, subset->GetBinLowEdge(1),  subset->GetBinLowEdge(bins) + subset->GetBinWidth(bins) );
  for(int i = 1; i<=bins; i++) {
    const double s = subset->GetBinContent(i);
    const double T = total->GetBinContent(i);
    const double B = T-s;

    const double p = T? s/T : 0;
    const double perr = T? ( (s&&B)? sqrt(s*s*B+B*B*s)/(T*T) : 1/T ) : 0;

    prob->SetBinContent(i,p);
    prob->SetBinError(i,perr);
  }  
  return prob;
}

LA_Filler_Fitter::Result LA_Filler_Fitter::
result(Method m, const std::string name, const Book& book) {
  Result p;
  const std::string base = boost::erase_all_copy(name,method(m));
  
  const TH1* const h = book[name];
  const TH1* const reco = book[base+"_reconstruction"];
  const TH1* const field = book[base+"_field"];

  if(reco) {
    p.reco    = reco->GetMean();
    p.recoErr = reco->GetRMS();
  }
  if(field) p.field = field->GetMean();

  if(h) {
    p.entries = (unsigned)(h->GetEntries());
    switch(m) {
    case WIDTH: {
      const TF1*const f = h->GetFunction("LA_profile_fit"); if(!f) break;
      p.measure = f->GetParameter(0);
      p.measureErr = f->GetParError(0);
      p.chi2 = f->GetChisquare();
      p.ndof = f->GetNDF();
      break;
    }
    case PROB1: case AVGV2: case AVGV3: case RMSV2: case RMSV3: /*case MULTI:*/ {
      const TF1*const f = h->GetFunction("SymmetryFit"); if(!f) break;
      p.measure = p.reco + f->GetParameter(0);
      p.measureErr = f->GetParameter(1);
      p.chi2 = f->GetParameter(2);
      p.ndof = (unsigned) (f->GetParameter(3));

      p.entries = 
	(m&PROB1)         ? (unsigned) book[base+"_w1"]->GetEntries() : 
	(m&(AVGV2|RMSV2)) ? (unsigned) book[base+method(AVGV2,0)]->GetEntries() : 
	(m&(AVGV3|RMSV3)) ? (unsigned) book[base+method(AVGV3,0)]->GetEntries() : 0 ;
      break;
    }
    default: break;
    }
  }
  return p;
}

std::map<uint32_t,LA_Filler_Fitter::Result> LA_Filler_Fitter::
module_results( const Book& book, const Method m) {
  std::map<uint32_t,Result> results;
  for(Book::const_iterator it = book.begin(".*_module\\d*"+method(m)); it!=book.end(); ++it ) {
    const uint32_t detid = boost::lexical_cast<uint32_t>( boost::regex_replace( it->first,
										boost::regex(".*_module(\\d*)_.*"),
										std::string("\\1")));
    results[detid] = result(m,it->first,book);
  }
  return results;
}

std::map<std::string,LA_Filler_Fitter::Result> LA_Filler_Fitter::
layer_results( const Book& book, const Method m) {
  std::map<std::string,Result> results;
  for(Book::const_iterator it = book.begin(".*layer\\d.*"+method(m)); it!=book.end(); ++it ) {
    const std::string name = boost::erase_all_copy(it->first,method(m));
    results[name] = result(m,it->first,book);
  }
  return results;
}

std::map<std::string, std::vector<LA_Filler_Fitter::Result> > LA_Filler_Fitter::
ensemble_results( const Book& book, const Method m) {
  std::map<std::string, std::vector<Result> > results;
  for(Book::const_iterator it = book.begin(".*_sample.*"+method(m)); it!=book.end(); ++it ) {
    const std::string name = boost::regex_replace(it->first,boost::regex("sample\\d*_"),"");
    results[name].push_back(result(m,it->first,book));
  }
  return results;
}

void LA_Filler_Fitter::
summarize_ensembles(Book& book) {
  typedef std::map<std::string, std::vector<Result> > results_t;
  results_t results;
  for(int m = FIRST_METHOD; m <= LAST_METHOD; m<<=1)
    if(methods_ & m) { results_t g = ensemble_results(book,(Method)(m)); results.insert(g.begin(),g.end());}
  
  BOOST_FOREACH(const results_t::value_type group, results) {
    const std::string name = group.first;
    BOOST_FOREACH(const Result p, group.second) {
      const float pad = (ensembleUp_-ensembleLow_)/10;
      book.fill( p.reco,       name+"_ensembleReco", 12*ensembleBins_, ensembleLow_-pad, ensembleUp_+pad );
      book.fill( p.measure,    name+"_measure",      12*ensembleBins_, ensembleLow_-pad, ensembleUp_+pad );
      book.fill( p.measureErr, name+"_merr",         500, 0, 0.01);
      book.fill( (p.measure-p.reco)/p.measureErr, name+"_pull", 500, -10,10);
    }
    book[name+"_measure"]->Fit("gaus","LLQ");
    book[name+"_merr"]->Fit("gaus","LLQ");
    book[name+"_pull"]->Fit("gaus","LLQ");
  }
}

std::map<std::string, std::vector<LA_Filler_Fitter::EnsembleSummary> > LA_Filler_Fitter::
ensemble_summary(const Book& book) {
  std::map<std::string, std::vector<EnsembleSummary> > summary;
  for(Book::const_iterator it = book.begin(".*_ensembleReco"); it!=book.end(); ++it) {
    const std::string base = boost::erase_all_copy(it->first,"_ensembleReco");

    const TH1*const reco = it->second;
    const TH1*const measure = book[base+"_measure"];
    const TH1*const merr = book[base+"_merr"];
    const TH1*const pull = book[base+"_pull"];

    EnsembleSummary s;
    s.truth = reco->GetMean();
    s.meanMeasured = measure->GetFunction("gaus")->GetParameter(1);
    s.SDmeanMeasured = measure->GetFunction("gaus")->GetParError(1);
    s.sigmaMeasured = measure->GetFunction("gaus")->GetParameter(2);
    s.SDsigmaMeasured = measure->GetFunction("gaus")->GetParError(2);
    s.meanUncertainty = merr->GetFunction("gaus")->GetParameter(1);
    s.SDmeanUncertainty = merr->GetFunction("gaus")->GetParError(1);
    s.pull = pull->GetFunction("gaus")->GetParameter(2);
    s.SDpull = pull->GetFunction("gaus")->GetParError(2);

    const std::string name = boost::regex_replace(base,boost::regex("ensembleBin\\d*_"),"");
    summary[name].push_back(s);
  }
  return summary;
}

std::pair<std::pair<float,float>, std::pair<float,float> > LA_Filler_Fitter::
offset_slope(const std::vector<LA_Filler_Fitter::EnsembleSummary>& ensembles) { 
  try {
    std::vector<float> x,y,xerr,yerr;
    BOOST_FOREACH(EnsembleSummary ensemble, ensembles) {
      x.push_back(ensemble.truth);
      xerr.push_back(0);
      y.push_back(ensemble.meanMeasured);
      yerr.push_back(ensemble.SDmeanMeasured);
    }
    TGraphErrors graph(x.size(),&(x[0]),&(y[0]),&(xerr[0]),&(yerr[0]));
    graph.Fit("pol1");
    const TF1*const fit = graph.GetFunction("pol1");
    
    return std::make_pair( std::make_pair(fit->GetParameter(0), fit->GetParError(0)),
			   std::make_pair(fit->GetParameter(1), fit->GetParError(1)) );
  } catch(edm::Exception e) { 
    std::cerr << "Fitting Line Failed " << std::endl << e << std::endl;
    return std::make_pair( std::make_pair(0,0), std::make_pair(0,0));
  }
}

float LA_Filler_Fitter::
pull(const std::vector<LA_Filler_Fitter::EnsembleSummary>& ensembles) {
  float p(0),w(0);
  BOOST_FOREACH(EnsembleSummary ensemble, ensembles) {
    const float unc2 = pow(ensemble.SDpull,2);
    p+=  ensemble.pull / unc2;
    w+= 1/unc2;
  }
  return p/w;
}


std::ostream& operator<<(std::ostream& strm, const LA_Filler_Fitter::Result& r) { 
  return strm << r.reco    <<"\t"<< r.recoErr <<"\t"
	      << r.measure <<"\t"<< r.measureErr <<"\t"
	      << r.calibratedMeasurement <<"\t"<< r.calibratedError <<"\t"
	      << r.field <<"\t"
	      << r.chi2 <<"\t"
	      << r.ndof <<"\t"
	      << r.entries;
}

std::ostream& operator<<(std::ostream& strm, const LA_Filler_Fitter::EnsembleSummary& e) { 
  return strm << e.truth <<"\t"
	      << e.meanMeasured    <<"\t"<< e.SDmeanMeasured <<"\t"
	      << e.sigmaMeasured   <<"\t"<< e.SDsigmaMeasured <<"\t"
	      << e.meanUncertainty <<"\t"<< e.SDmeanUncertainty << "\t"
	      << e.pull            <<"\t"<< e.SDpull;
}



