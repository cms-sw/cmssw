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

void LA_Filler_Fitter::
fill(TTree* tree, Book& book) {
  TTREE_FOREACH_ENTRY(tree,
		      std::vector<unsigned>* LEAF( clusterdetid )
		      std::vector<unsigned>* LEAF( clusterwidth )
		      std::vector<float>*    LEAF( clustervariance )
		      std::vector<unsigned>* LEAF( tsostrackmulti )
		      std::vector<float>*    LEAF( tsosdriftx )
		      std::vector<float>*    LEAF( tsosdriftz )
		      std::vector<float>*    LEAF( tsoslocalpitch )
		      std::vector<float>*    LEAF( tsoslocaltheta )
		      std::vector<float>*    LEAF( tsoslocalphi )
		      std::vector<float>*    LEAF( tsosBdotY )
		      std::vector<float>*    LEAF( tsosglobalZofunitlocalY )
		      ) {
    if(maxEvents_) TTREE_FOREACH_ENTRY_total = std::min(maxEvents_,TTREE_FOREACH_ENTRY_total);
    for(unsigned i=0; i< clusterwidth->size() ; i++) {  

      SiStripDetId detid((*clusterdetid)[i]);

      if( (*tsostrackmulti)[i] != 1 ||
	  detid.subDetector()!=SiStripDetId::TIB && 
	  detid.subDetector()!=SiStripDetId::TOB        ) 
	continue;
      
      double tthetaL = (*tsosdriftx)[i] / (*tsosdriftz)[i];
      double tthetaT = tan((*tsoslocaltheta)[i]) * cos((*tsoslocalphi)[i]);
      int sign = (*tsosglobalZofunitlocalY)[i] < 0  ?  -1  :  1 ;
      unsigned width = (*clusterwidth)[i];
      double var = (*clustervariance)[i];

      poly<std::string> granular;
      if(ensembleBins_) {
	granular+= "ensembleBin"+boost::lexical_cast<std::string>((int)(ensembleBins_*(sign*tthetaL-ensembleLow_)/(ensembleUp_-ensembleLow_)));
	granular+= "_pitch"+boost::lexical_cast<std::string>((int)(10000* (*tsoslocalpitch)[i] ));
	granular+= "";
	if(ensembleSize_) granular*= "_sample"+boost::lexical_cast<std::string>(TTREE_FOREACH_ENTRY_index % ensembleSize_);
      } else {
	granular += subdetLabel(detid);
	if(byLayer_) granular *= layerLabel(detid);
	if(byModule_) granular *= moduleLabel(detid);
      }

      poly<std::string> W; W++;      
      poly<std::string> A1("_all");  if(width==1) A1*="_w1"; 
                                else if(width==2)  W*="_w2"; 
				else if(width==3)  W*="_w3";

                             book.fill( sign*tthetaL,                    granular+"_reconstruction"   , 81, -0.6,0.6 );
      if(methods_ & RATIO)   book.fill( sign*tthetaT,                    granular+ method(RATIO,0)+A1 , 81, -0.6,0.6 );
      if(methods_ & WIDTH)   book.fill( sign*tthetaT,             width, granular+ method(WIDTH)      , 81, -0.6,0.6 );
      if(methods_ & SQRTVAR) book.fill( sign*tthetaT,         sqrt(var), granular+ method(SQRTVAR)    , 81, -0.6,0.6 );
      if(methods_ & SYMM)    book.fill( sign*tthetaT,         sqrt(var), granular+ method(SYMM,0)     ,360, -1.0,1.0 );

      if(methods_ & MULTI){  book.fill( sign*(tthetaT-tthetaL),     var, granular+method(MULTI,0)+"_var" +W ,360,-1.0,1.0 );
	                     book.fill( sign*(tthetaT-tthetaL), var*var, granular+method(MULTI,0)+"_var2"+W ,360,-1.0,1.0 );
			     book.fill( sign*(tthetaT-tthetaL),          granular+ method(MULTI,0)+A1       ,360,-1.0,1.0 ); }

      if(ensembleBins_==0)   book.fill( fabs((*tsosBdotY)[i]),           granular+"_field"            , 101, 1, 5 );
    }
  }
}

std::string LA_Filler_Fitter::
subdetLabel(SiStripDetId detid) { return detid.subDetector()==SiStripDetId::TOB? "TOB" : "TIB";}
std::string LA_Filler_Fitter::
moduleLabel(SiStripDetId detid) { return subdetLabel(detid) + "_module"+boost::lexical_cast<std::string>(detid());}
std::string LA_Filler_Fitter::
layerLabel(SiStripDetId detid) {
  unsigned layer = detid.subDetector() == SiStripDetId::TOB ? TOBDetId(detid()).layer() : TIBDetId(detid()).layer();
  return subdetLabel(detid)+"_layer"+boost::lexical_cast<std::string>(layer)+(detid.stereo()?"s":"a");
}

void LA_Filler_Fitter::
make_and_fit_ratio(Book& book, bool cleanup) {
  for(Book::const_iterator it = book.begin(".*"+method(RATIO,0)+"_w1"); it!=book.end(); ++it) {
    if((*it)->GetEntries() < 30) continue;

    std::string base = boost::erase_all_copy(it.name(),"_w1");
    std::string width1 = base+"_w1";
    std::string all    = base+"_all";
    std::string ratio  = base+"_ratio";

    TH1* p = (TH1*) book(width1)->Clone(ratio.c_str());
    p->Divide(book(all));
    p->Fit("gaus","LLQ");
    double mean = p->GetFunction("gaus")->GetParameter(1);
    double sigma = p->GetFunction("gaus")->GetParameter(2);
    p->Fit("gaus","Q","",mean-sigma,mean+sigma);
    p->SetTitle("Ratio Method;tan#theta_{t};Probability of width==1");
    book.book(ratio, p);
    
    if(cleanup) {
      book.erase(width1);
      book.erase(all);
    }
  }
}

void LA_Filler_Fitter::
fit_profile(Book& book, const std::string& key) {
  for(Book::const_iterator p = book.begin(".*"+key); p!=book.end(); ++p) {
    if((*p)->GetEntries() < 400) {book.erase(p.name()); continue;}
    (*p)->SetTitle(";tan#theta_{t};");
    float min = (*p)->GetMinimum();
    float max = (*p)->GetMaximum();
    float xofmin = (*p)->GetBinCenter((*p)->GetMinimumBin()); if( xofmin>0.0 || xofmin<-0.15) xofmin = -0.05;
    float xofmax = (*p)->GetBinCenter((*p)->GetMaximumBin());

    TF1* fit = new TF1("LA_profile_fit","[2]*(TMath::Abs(x-[0]))+[1]",-1,1);
    fit->SetParLimits(0,-0.15,0.01);
    fit->SetParLimits(1, 0.6*min, 1.25*min );
    fit->SetParLimits(2,0.1,10);
    fit->SetParameters( xofmin, min, (max-min) / fabs( xofmax - xofmin ) );

    int badfit = (*p)->Fit(fit,"IEQ","",-.5,.3);
    if( badfit ) badfit = (*p)->Fit(fit,"IEQ","", -.46,.26);
    if( badfit ) book.erase(p.name());
  }
}

void LA_Filler_Fitter::
make_and_fit_symmchi2(Book& book) {
  for(Book::const_iterator p = book.begin(".*"+method(SYMM,0)); p!=book.end(); ++p) {
    (*p)->SetTitle(";tan#theta_{t};mean sqrt(variance)");
    unsigned rebin = (unsigned)( (*p)->GetNbinsX() / sqrt(2*(*p)->GetEntries()) + 1);
    (*p)->Rebin( rebin>1 ? rebin<7 ? rebin : 6 : 1);
    
    const unsigned bins = (*p)->GetNbinsX();
    const unsigned guess = guess_bin(*p);
    TH1* chi2 = SymmetryFit::symmetryChi2(*p, std::make_pair(guess-bins/20,guess+bins/20));
    if(chi2) { 
      const unsigned guess2 = (*p)->FindBin(chi2->GetFunction("SymmetryFit")->GetParameter(0));
      delete chi2;
      chi2 = SymmetryFit::symmetryChi2(*p, std::make_pair(guess2-bins/30,guess2+bins/30));
      if(chi2) book.book(SymmetryFit::name((*p)->GetName()), chi2);
    }
  }
}

unsigned LA_Filler_Fitter::
guess_bin(TH1* hist) {
  const unsigned bins = hist->GetNbinsX();
  unsigned lower(1), upper(1+bins/20);  double sliding_sum(0);
  for(unsigned bin=lower; bin<=upper; ++bin) {double c = hist->GetBinContent(bin); if(c==0) upper++; sliding_sum+=c;}
  unsigned least_low(lower), least_up(upper); double least_sum(sliding_sum);
  while(upper<=bins) {
    while( ++upper<=bins && hist->GetBinContent(upper) == 0 ); 
    if(upper<=bins) {
      while( hist->GetBinContent(++lower) == 0 );
      sliding_sum += hist->GetBinContent(upper);
      sliding_sum -= hist->GetBinContent(lower);
      if( sliding_sum < least_sum ) {
	least_sum = sliding_sum;
	least_low = lower;
	least_up = upper;
      }
    }
  }
  return (least_low+least_up)/2;
}


void LA_Filler_Fitter::
make_and_fit_multisymmchi2(Book& book) {
  for(Book::const_iterator it = book.begin(".*"+method(MULTI,0)+"_var_w2"); it!=book.end(); ++it) {
    std::string base = boost::erase_all_copy(it.name(),"_var_w2");

    std::vector<TH1*> rebin_hists;
    TH1* prob_w1 = book(base+"_w1");     rebin_hists.push_back(prob_w1);
    TH1* all = book(base+"_all");        rebin_hists.push_back(all);
    TH1* var_w2 = book(base+"_var_w2");   rebin_hists.push_back(var_w2);
    TH1* var_w3 = book(base+"_var_w3");   rebin_hists.push_back(var_w3);
    TH1* var2_w2 = book(base+"_var2_w2"); rebin_hists.push_back(var2_w2);
    TH1* var2_w3 = book(base+"_var2_w3"); rebin_hists.push_back(var2_w3);

    unsigned rebin = std::max(find_rebin(var_w2),find_rebin(var_w3));
    BOOST_FOREACH(TH1* hist, rebin_hists) hist->Rebin( rebin>1 ? rebin<7 ? rebin : 6 : 1);

    TH1* rmsv_w2 = rms_from_x_xx(base+"_rms_w2",var_w2,var2_w2); book.book(base+"_rms_w2",rmsv_w2);
    TH1* rmsv_w3 = rms_from_x_xx(base+"_rms_w3",var_w3,var2_w3); book.book(base+"_rms_w3",rmsv_w3);
    prob_w1->Divide(all);
    for(int i=1; i<=prob_w1->GetNbinsX(); i++) if(!prob_w1->GetBinError(i)) prob_w1->SetBinError(i,1);

    book.erase(base+"_var2_w2");
    book.erase(base+"_var2_w3");
    book.erase(base+"_all");

    std::vector<TH1*> fit_hists;
    fit_hists.push_back(prob_w1);  prob_w1->SetTitle("Width==1 Probability;tan#theta_{t}-(dx/dz)_{reco}");
    fit_hists.push_back(var_w2);   var_w2->SetTitle("Width==2 Mean Variance;tan#theta_{t}-(dx/dz)_{reco}");
    fit_hists.push_back(var_w3);   var_w3->SetTitle("Width==3 Mean Variance;tan#theta_{t}-(dx/dz)_{reco}");
    fit_hists.push_back(rmsv_w2);  rmsv_w2->SetTitle("Width==2 RMS Variance;tan#theta_{t}-(dx/dz)_{reco}");
    fit_hists.push_back(rmsv_w3);  rmsv_w3->SetTitle("Width==3 RMS Variance;tan#theta_{t}-(dx/dz)_{reco}");

    const unsigned bins = fit_hists[0]->GetNbinsX();
    const unsigned guess = fit_hists[0]->FindBin(0);
    TH1* chi2 = SymmetryFit::symmetryChi2(base, fit_hists, std::make_pair(guess-bins/30, guess+bins/30-1));
    if(chi2) { book.book(SymmetryFit::name(base), chi2); chi2->SetTitle("MultiSymmetry #chi^{2};tan#theta_{t}-(dx/dz)_{reco}");}
  }
}

unsigned LA_Filler_Fitter::find_rebin(TH1* hist) {
  double mean = hist->GetMean();
  double rms = hist->GetRMS();
  int begin = std::min(                1, hist->FindBin(mean-rms));
  int end   = std::max(hist->GetNbinsX(), hist->FindBin(mean+rms)) + 1;
  unsigned current_hole(0), max_hole(0);
  for(int i=begin; i<end; i++) {
    if(!hist->GetBinError(i)) current_hole++;
    else if(current_hole) {max_hole = std::max(current_hole,max_hole); current_hole=0;}
  }
  return max_hole+1;
}

TH1* LA_Filler_Fitter::rms_from_x_xx(std::string name, TH1* m, TH1* mm) {
  int bins = m->GetNbinsX();
  TH1* rms = new TH1F(name.c_str(),"",bins, m->GetBinLowEdge(1),  m->GetBinLowEdge(bins) + m->GetBinWidth(bins) );
  for(int i = 1; i<=bins; i++) {
    double M = m->GetBinContent(i);
    double MM = mm->GetBinContent(i);
    double Me = m->GetBinError(i);
    double MMe = mm->GetBinError(i);

    if(Me) {
      rms->SetBinContent(i, sqrt( MM - M*M ) );
      rms->SetBinError(i, sqrt( 0.5*MMe*MMe + Me*Me ) );
    }
  }
  return rms;
}

LA_Filler_Fitter::Result LA_Filler_Fitter::
result(Method m, const std::string name, const Book& book) {
  Result p;
  std::string base = boost::erase_all_copy(name,method(m));
  if(book.contains(base+"_reconstruction")) {
    p.reco    = book(base+"_reconstruction")->GetMean();
    p.recoErr = book(base+"_reconstruction")->GetRMS();
  }
  if(book.contains(base+"_field"))
    p.field = book(base+"_field")->GetMean();

  if(book.contains(name)) {
    TH1* h = book(name);
    p.entries = (unsigned)(h->GetEntries());
    switch(m) {
    case RATIO: {
      TF1* f = h->GetFunction("gaus"); if(!f) break;
      p.measure = f->GetParameter(1);
      p.measureErr = f->GetParError(1);
      p.chi2 = f->GetChisquare();
      p.ndof = f->GetNDF();
      break; }
    case WIDTH: case SQRTVAR: {
      TF1* f = h->GetFunction("LA_profile_fit"); if(!f) break;
      p.measure = f->GetParameter(0);
      p.measureErr = f->GetParError(0);
      p.chi2 = f->GetChisquare();
      p.ndof = f->GetNDF();
      break;
    }
    case SYMM: {
      p.entries = (unsigned) book(base+method(SYMM,0))->GetEntries();
      TF1* f = h->GetFunction("SymmetryFit"); if(!f) break;
      p.measure = f->GetParameter(0);
      p.measureErr = f->GetParameter(1);
      p.chi2 = f->GetParameter(2);
      p.ndof = (unsigned) (f->GetParameter(3));
      break;
    }
    case MULTI: {
      p.entries = (unsigned) book(base+method(MULTI,0)+"_w1")->GetEntries();
      p.entries+= (unsigned) book(base+method(MULTI,0)+"_var_w2")->GetEntries();
      p.entries+= (unsigned) book(base+method(MULTI,0)+"_var_w3")->GetEntries();
      TF1* f = h->GetFunction("SymmetryFit"); if(!f) break;
      p.measure = p.reco + f->GetParameter(0);
      p.measureErr = f->GetParameter(1);
      p.chi2 = f->GetParameter(2);
      p.ndof = (unsigned) (f->GetParameter(3));
      break;
    }
    default:break;
    }
  }
  return p;
}

std::map<uint32_t,LA_Filler_Fitter::Result> LA_Filler_Fitter::
module_results(const Book& book, const Method m) {
  std::map<uint32_t,Result> results;
  for(Book::const_iterator it = book.begin(".*_module\\d*"+method(m)); it!=book.end(); ++it ) {
    uint32_t detid = boost::lexical_cast<uint32_t>( boost::regex_replace( it.name(),
									  boost::regex(".*_module(\\d*)_.*"),
									  std::string("\\1")));
    results[detid] = result(m,it.name(),book);
  }
  return results;
}

std::map<std::string,LA_Filler_Fitter::Result> LA_Filler_Fitter::
layer_results(const Book& book, const Method m) {
  std::map<std::string,Result> results;
  for(Book::const_iterator it = book.begin(".*layer\\d.*"+method(m)); it!=book.end(); ++it ) {
    std::string name = boost::erase_all_copy(it.name(),method(m));
    results[name] = result(m,it.name(),book);
  }
  return results;
}

std::map<std::string, std::vector<LA_Filler_Fitter::Result> > LA_Filler_Fitter::
ensemble_results(const Book& book, const Method m) {
  std::map<std::string, std::vector<Result> > results;
  for(Book::const_iterator it = book.begin(".*_sample.*"+method(m)); it!=book.end(); ++it ) {
    std::string name = boost::regex_replace(it.name(),boost::regex("sample\\d*_"),"");
    results[name].push_back(result(m,it.name(),book));
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
    try {
      BOOST_FOREACH(const Result p, group.second) {
	float pad = (ensembleUp_-ensembleLow_)/10;
	book.fill( p.reco,       name+"_ensembleReco", 12*ensembleBins_, ensembleLow_-pad, ensembleUp_+pad );
	book.fill( p.measure,    name+"_measure",      12*ensembleBins_, ensembleLow_-pad, ensembleUp_+pad );
	book.fill( p.measureErr, name+"_merr",         500, 0, 0.1);
	book.fill( (p.measure-p.reco)/p.measureErr, name+"_pull", 500, -10,10);
      }
      book(name+"_measure")->Fit("gaus","LLQ");
      book(name+"_merr")->Fit("gaus","LLQ");
      book(name+"_pull")->Fit("gaus","LLQ");
    } catch (edm::Exception e) {
      std::cerr << "Fit summary failed " << std::endl << e << std::endl;
      book.erase(name+"_ensembleReco");
      book.erase(name+"_measure");
      book.erase(name+"_merr");
      book.erase(name+"_pull");
    }
  }
}

std::map<std::string, std::vector<LA_Filler_Fitter::EnsembleSummary> > LA_Filler_Fitter::
ensemble_summary(const Book& book) {
  std::map<std::string, std::vector<EnsembleSummary> > summary;
  for(Book::const_iterator it = book.begin(".*_ensembleReco"); it!=book.end(); ++it) {
    std::string base = boost::erase_all_copy(it.name(),"_ensembleReco");

    TH1* reco = *it;
    TH1* measure = book(base+"_measure");
    TH1* merr = book(base+"_merr");
    TH1* pull = book(base+"_pull");

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

    std::string name = boost::regex_replace(base,boost::regex("ensembleBin\\d*_"),"");
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
    TF1* fit = graph.GetFunction("pol1");
    
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
    float unc2 = pow(ensemble.SDpull,2);
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



