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
		      const std::vector<unsigned>* LEAF( clusterdetid )
		      const std::vector<unsigned>* LEAF( clusterwidth )
		      const std::vector<float>*    LEAF( clustervariance )
		      const std::vector<unsigned>* LEAF( tsostrackmulti )
		      const std::vector<float>*    LEAF( tsosdriftx )
		      const std::vector<float>*    LEAF( tsosdriftz )
		      const std::vector<float>*    LEAF( tsoslocalpitch )
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
      if(ensembleBins_) {
	granular+= "_ensembleBin"+boost::lexical_cast<std::string>((int)(ensembleBins_*(sign*tthetaL-ensembleLow_)/(ensembleUp_-ensembleLow_)));
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
subdetLabel(const SiStripDetId detid) { return detid.subDetector()==SiStripDetId::TOB? "TOB" : "TIB";}
std::string LA_Filler_Fitter::
moduleLabel(const SiStripDetId detid) { return subdetLabel(detid) + "_module"+boost::lexical_cast<std::string>(detid());}
std::string LA_Filler_Fitter::
layerLabel(const SiStripDetId detid) {
  unsigned layer = detid.subDetector() == SiStripDetId::TOB ? TOBDetId(detid()).layer() : TIBDetId(detid()).layer();
  return subdetLabel(detid)+"_layer"+boost::lexical_cast<std::string>(layer)+(detid.stereo()?"s":"a");
}

void LA_Filler_Fitter::
make_and_fit_ratio(Book& book) {
  for(Book::iterator it=book.begin(".*"+method(RATIO,0)+"_w1"); it!=book.end(); book.erase(it++)) {
    if((it->second)->GetEntries() < 30) continue;

    const std::string base = boost::erase_all_copy(it->first,"_w1");
    TH1* const p = subset_probability(base+"_ratio", it->second, book[base+"_all"] );

    p->Fit("gaus","LLQ");
    const double mean = p->GetFunction("gaus")->GetParameter(1);
    const double sigma = p->GetFunction("gaus")->GetParameter(2);
    p->Fit("gaus","Q","",mean-sigma,mean+sigma);
    p->SetTitle("Ratio Method;tan#theta_{t};Probability of width==1");
    book.book(base+"_ratio", p);

    book.erase(base+"_all");
  }
}

void LA_Filler_Fitter::
fit_profile(Book& book, const std::string key) {
  for(Book::iterator it = book.begin(".*"+key); it!=book.end(); ++it) {
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
  for(Book::iterator it = book.begin(".*"+method(SYMM,0)); it!=book.end(); ++it) {
    TH1* const p = it->second;
    p->SetTitle(";tan#theta_{t};mean sqrt(variance)");
    const unsigned rebin = (unsigned)( p->GetNbinsX() / sqrt(2*p->GetEntries()) + 1);
    p->Rebin( rebin>1 ? rebin<7 ? rebin : 6 : 1);
    
    const unsigned bins = p->GetNbinsX();
    const unsigned guess = guess_bin(p);
    TH1* chi2 = SymmetryFit::symmetryChi2(p, std::make_pair(guess-bins/20,guess+bins/20));
    if(chi2) { 
      const unsigned guess2 = p->FindBin(chi2->GetFunction("SymmetryFit")->GetParameter(0));
      delete chi2;
      chi2 = SymmetryFit::symmetryChi2(p, std::make_pair(guess2-bins/30,guess2+bins/30));
      if(chi2) book.book(SymmetryFit::name(p->GetName()), chi2);
    }
  }
}

unsigned LA_Filler_Fitter::
guess_bin(const TH1* const hist) {
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
  for(Book::iterator it = book.begin(".*"+method(MULTI,0)+"_var_w2"); it!=book.end(); ++it) {
    const std::string base = boost::erase_all_copy(it->first,"_var_w2");

    std::vector<TH1*> rebin_hists;
    TH1* const w1 = book[base+"_w1"];           rebin_hists.push_back(w1);
    TH1* const all = book[base+"_all"];         rebin_hists.push_back(all);
    TH1* const var_w2 = book[base+"_var_w2"];   rebin_hists.push_back(var_w2);
    TH1* const var_w3 = book[base+"_var_w3"];   rebin_hists.push_back(var_w3);
    TH1* const var2_w2 = book[base+"_var2_w2"]; rebin_hists.push_back(var2_w2);
    TH1* const var2_w3 = book[base+"_var2_w3"]; rebin_hists.push_back(var2_w3);

    const unsigned rebin = std::max(find_rebin(var_w2),find_rebin(var_w3));
    BOOST_FOREACH(TH1*const hist, rebin_hists) hist->Rebin( rebin>1 ? rebin<7 ? rebin : 6 : 1);

    TH1* const prob_w1 = subset_probability(base+"_prob_w1",w1,all);   book.book(base+"_prob_w1",prob_w1);
    TH1* const rmsv_w2 = rms_from_x_xx(base+"_rms_w2",var_w2,var2_w2); book.book(base+"_rms_w2",rmsv_w2);
    TH1* const rmsv_w3 = rms_from_x_xx(base+"_rms_w3",var_w3,var2_w3); book.book(base+"_rms_w3",rmsv_w3);
    
    book.erase(base+"_all");
    book.erase(base+"_var2_w2");
    book.erase(base+"_var2_w3");
    book.erase(base+"_w1");

    std::vector<TH1*> fit_hists;
    fit_hists.push_back(prob_w1);  prob_w1->SetTitle("Width==1 Probability;tan#theta_{t}-(dx/dz)_{reco}");
    fit_hists.push_back(var_w2);   var_w2->SetTitle("Width==2 Mean Variance;tan#theta_{t}-(dx/dz)_{reco}");
    fit_hists.push_back(var_w3);   var_w3->SetTitle("Width==3 Mean Variance;tan#theta_{t}-(dx/dz)_{reco}");
    fit_hists.push_back(rmsv_w2);  rmsv_w2->SetTitle("Width==2 RMS Variance;tan#theta_{t}-(dx/dz)_{reco}");
    fit_hists.push_back(rmsv_w3);  rmsv_w3->SetTitle("Width==3 RMS Variance;tan#theta_{t}-(dx/dz)_{reco}");

    const unsigned bins = fit_hists[0]->GetNbinsX();
    const unsigned guess = fit_hists[0]->FindBin(0);
    TH1* const chi2 = SymmetryFit::symmetryChi2(base, fit_hists, std::make_pair(guess-bins/30, guess+bins/30-1));
    if(chi2) { book.book(SymmetryFit::name(base), chi2); chi2->SetTitle("MultiSymmetry #chi^{2};tan#theta_{t}-(dx/dz)_{reco}");}
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
rms_from_x_xx(const std::string name, const TH1* const m, const TH1* const mm) {
  const int bins = m->GetNbinsX();
  TH1* const rms = new TH1F(name.c_str(),"",bins, m->GetBinLowEdge(1),  m->GetBinLowEdge(bins) + m->GetBinWidth(bins) );
  for(int i = 1; i<=bins; i++) {
    const double M = m->GetBinContent(i);
    const double MM = mm->GetBinContent(i);
    const double Me = m->GetBinError(i);
    const double MMe = mm->GetBinError(i);

    if(Me) {
      rms->SetBinContent(i, sqrt( MM - M*M ) );
      rms->SetBinError(i, sqrt( 0.5*MMe*MMe + Me*Me ) );
    }
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
    case RATIO: {
      const TF1*const f = h->GetFunction("gaus"); if(!f) break;
      p.measure = f->GetParameter(1);
      p.measureErr = f->GetParError(1);
      p.chi2 = f->GetChisquare();
      p.ndof = f->GetNDF();
      break; }
    case WIDTH: case SQRTVAR: {
      const TF1*const f = h->GetFunction("LA_profile_fit"); if(!f) break;
      p.measure = f->GetParameter(0);
      p.measureErr = f->GetParError(0);
      p.chi2 = f->GetChisquare();
      p.ndof = f->GetNDF();
      break;
    }
    case SYMM: {
      p.entries = (unsigned) book[base+method(SYMM,0)]->GetEntries();
      const TF1*const f = h->GetFunction("SymmetryFit"); if(!f) break;
      p.measure = f->GetParameter(0);
      p.measureErr = f->GetParameter(1);
      p.chi2 = f->GetParameter(2);
      p.ndof = (unsigned) (f->GetParameter(3));
      break;
    }
    case MULTI: {
      p.entries = (unsigned) book[base+method(MULTI,0)+"_prob_w1"]->GetEntries();
      p.entries+= (unsigned) book[base+method(MULTI,0)+"_var_w2"]->GetEntries();
      p.entries+= (unsigned) book[base+method(MULTI,0)+"_var_w3"]->GetEntries();
      const TF1*const f = h->GetFunction("SymmetryFit"); if(!f) break;
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



