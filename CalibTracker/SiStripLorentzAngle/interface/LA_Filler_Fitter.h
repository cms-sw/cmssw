#ifndef LA_FILLER_FITTER_H
#define LA_FILLER_FITTER_H

#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <boost/lexical_cast.hpp>
#include <TTree.h>
#include <TF1.h>
#include <TGraphErrors.h>
#include "TTREE_FOREACH_ENTRY.hh"
#include "Book.h"

#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"

class LA_Filler_Fitter {

 public:

  static LA_Filler_Fitter byRecoLA(int M, int N, double low, double up) {return LA_Filler_Fitter(M,N,low,up,false,false,false,true,true);}
  static LA_Filler_Fitter bySubdetector() {return LA_Filler_Fitter(0,0,0,0,true,false,false,false,true);}
  static LA_Filler_Fitter bySubdetectorLayer() {return LA_Filler_Fitter(0,0,0,0,true,true,false,false,true);}
  static LA_Filler_Fitter bySubdetectorLayerModule() {return LA_Filler_Fitter(0,0,0,0,true,true,true,false,true);}

  LA_Filler_Fitter& setMaxEvents(Long64_t max) {maxEvents=max; return *this;}
  LA_Filler_Fitter& fill(TTree*, Book&);
  static void fit(Book& book) { make_and_fit_ratio(book); make_and_fit_profile(book,"-tanLA"); }
  static void make_and_fit_ratio(Book&);
  static void make_and_fit_profile(Book&, const std::string&);

  struct Result { 
    float reco,measure,recoErr,measureErr,chi2,field; 
    unsigned ndof,entries; 
    Result() : reco(0), measure(0), recoErr(0), measureErr(0), chi2(0), field(0), ndof(0), entries(0) {}
  };
  static std::map<std::string, std::vector<Result> > harvest_results(const Book&);
  static TGraphErrors* tgraphe(const std::vector<Result>&);
  static void print_to_file(std::string, const std::vector<Result>&);

 private:

  LA_Filler_Fitter(int M, int N, double low, double up, bool sub, bool layer, bool module, bool pitch, bool sum) : 
    ensemble_size(M),
    tanLA_bins(N),tanLA_low(low),tanLA_up(up),
    bySub(sub),byLayer(layer),byModule(module),byPitch(pitch),summary(sum),
    maxEvents(0)
    {};

  int ensemble_size, tanLA_bins;
  double tanLA_low, tanLA_up;
  bool bySub, byLayer, byModule, byPitch, summary;
  Long64_t maxEvents;

};

#endif

LA_Filler_Fitter& LA_Filler_Fitter::
fill(TTree* tree, Book& book) {
  TTREE_FOREACH_ENTRY(tree,
		      std::vector<unsigned>* LEAF( clusterdetid )
		      std::vector<unsigned>* LEAF( clusterwidth )
		      std::vector<float>*    LEAF( clustervariance )
		      std::vector<unsigned>* LEAF( tsostrackmulti )
		      std::vector<int>*      LEAF( tsostrackindex )
		      std::vector<float>*    LEAF( tsosdriftx )
		      std::vector<float>*    LEAF( tsosdriftz )
		      std::vector<float>*    LEAF( tsoslocalpitch )
		      std::vector<float>*    LEAF( tsoslocaltheta )
		      std::vector<float>*    LEAF( tsoslocalphi )
		      std::vector<float>*    LEAF( tsosBdotY )
		      std::vector<float>*    LEAF( trackchi2ndof )
		      std::vector<unsigned>* LEAF( trackhitsvalid )
		      ) {
    if(maxEvents) TTREE_FOREACH_ENTRY_total = std::min(maxEvents,TTREE_FOREACH_ENTRY_total);
    for(unsigned i=0; i< clusterwidth->size() ; i++) {  

      SiStripDetId detid((*clusterdetid)[i]);

      if( (*tsostrackmulti)[i] != 1 ||
	  (*trackchi2ndof)[(*tsostrackindex)[i]] > 10 ||
	  (*trackhitsvalid)[(*tsostrackindex)[i]] < 7 ||
	  detid.subDetector()!=SiStripDetId::TIB && 
	  detid.subDetector()!=SiStripDetId::TOB        ) 
	continue;
      
      float BdotY = (*tsosBdotY)[i];
      float driftx = (*tsosdriftx)[i];
      float driftz = (*tsosdriftz)[i];

      int sign = driftx < 0  ?  1  :  -1 ;
      double projection = driftz * tan((*tsoslocaltheta)[i]) * cos((*tsoslocalphi)[i]);
      unsigned width = (*clusterwidth)[i];
      float sigma = sqrt((*clustervariance)[i]);
      float pitch = (*tsoslocalpitch)[i];

      poly<std::string> granular;
      if(bySub) granular+= detid.subDetector()==SiStripDetId::TOB? "TOB" : "TIB";
      if(tanLA_bins) granular+= "recoLA"+boost::lexical_cast<std::string>((int)(tanLA_bins*(sign*driftx/driftz-tanLA_low)/(tanLA_up-tanLA_low)));
      if(byPitch) granular+= "_pitch"+boost::lexical_cast<std::string>((int)(pitch*10000));
      granular++;

      if(byLayer) {     unsigned layer = detid.subDetector() == SiStripDetId::TOB ? TOBDetId(detid()).layer() : TIBDetId(detid()).layer(); 
 	                granular*= "_layer"+boost::lexical_cast<std::string>(layer)+(detid.stereo()?"s":"a"); }
      if(byModule)      granular*= "_module"+boost::lexical_cast<std::string>(detid());
      if(ensemble_size)	granular*= "_sample"+boost::lexical_cast<std::string>(TTREE_FOREACH_ENTRY_index%ensemble_size);
      if(summary)       granular*= "";

      poly<std::string> all_one("_all"); 
      if(width==1) all_one*="_width1";

      float N=2.5;
      book.fill( sign*fabs(BdotY),              granular+"_field"          , 101, 1, 5 );
      book.fill( sign*driftx/driftz,            granular+"_reconstruction" , 101, -N*pitch/driftz, N*pitch/driftz          );
      book.fill( sign*projection/driftz,        granular+all_one+"_tanLA"  , 101, -N*pitch/driftz, N*pitch/driftz          );
      book.fill( sign*projection/driftz, width, granular+"_width-tanLA"    , 101, -N*pitch/driftz, N*pitch/driftz, 20,0,20 );
      book.fill( sign*projection/driftz, sigma, granular+"_variance-tanLA" , 101, -N*pitch/driftz, N*pitch/driftz, 100,0,2 );
    }
  }
  return *this;
}

void LA_Filler_Fitter::
make_and_fit_ratio(Book& book) {
  for(Book::const_iterator width1 = book.begin(".*width1.*"); width1!=book.end(); ++width1) {
    std::string all_name = width1.name();      all_name.replace( all_name.find("width1"),6,"all");
    std::string ratio_name = width1.name();  ratio_name.replace( ratio_name.find("width1"),6,"ratio");
    book.book(ratio_name, (TH1*) book(width1.name())->Clone(ratio_name.c_str()))->Divide(book(all_name));
    book(ratio_name)->Fit("gaus","Q");
    double mean = book(ratio_name)->GetFunction("gaus")->GetParameter(1);
    double sigma = book(ratio_name)->GetFunction("gaus")->GetParameter(2);
    book(ratio_name)->Fit("gaus","IMEQ","",mean-sigma,mean+sigma);
  }
}

void LA_Filler_Fitter::
make_and_fit_profile(Book& book, const std::string& key) {
  for(Book::const_iterator hist2D = book.begin(".*"+key); hist2D!=book.end(); ++hist2D) {
    std::string name = hist2D.name()+"_profile";
    TH1* p = book.book(name, (TH1*) ((TH2*)(*hist2D))->ProfileX(name.c_str()));
    float min = p->GetMinimum();
    float max = p->GetMaximum();
    float xofmin = p->GetBinCenter(p->GetMinimumBin()); if( xofmin>0.0 || xofmin<-0.15) xofmin = -0.05;
    float xofmax = p->GetBinCenter(p->GetMaximumBin());

    TF1* fit = new TF1("LA_profile_fit","[2]*(TMath::Abs(x-[0]))+[1]",-1,1);
    fit->SetParLimits(0,-0.15,0.01);
    fit->SetParLimits(1, 0.6*min, 1.25*min );
    fit->SetParLimits(2,0.1,10);
    fit->SetParameters( xofmin, min, (max-min) / fabs( xofmax - xofmin ) );
    p->Fit(fit,"IMEQ");
    if( p->GetFunction("LA_profile_fit")->GetChisquare() / p->GetFunction("LA_profile_fit")->GetNDF() > 5 ||
	p->GetFunction("LA_profile_fit")->GetParError(0) > 0.03) 
      p->Fit(fit,"IMEQ");
  }
}

std::map<std::string, std::vector<LA_Filler_Fitter::Result> > LA_Filler_Fitter::
harvest_results(const Book& book) {
  std::map<std::string, std::vector<Result> > harvest;
  for(Book::const_iterator sample = book.begin(".*_sample.*"); sample!=book.end(); ++sample ) {
    Result p;

    if( sample.name().find("ratio") != std::string::npos ) {
      p.measure = (*sample)->GetFunction("gaus")->GetParameter(1);
      p.measureErr = (*sample)->GetFunction("gaus")->GetParError(1);
      p.chi2 = (*sample)->GetFunction("gaus")->GetChisquare();
      p.ndof = (*sample)->GetFunction("gaus")->GetNDF();
    } else 
    if( sample.name().find("profile") != std::string::npos ) {
      p.measure = (*sample)->GetFunction("LA_profile_fit")->GetParameter(0);
      p.measureErr = (*sample)->GetFunction("LA_profile_fit")->GetParError(0);
      p.chi2 = (*sample)->GetFunction("LA_profile_fit")->GetChisquare();
      p.ndof = (*sample)->GetFunction("LA_profile_fit")->GetNDF();
    } else continue;

    std::string s = sample.name();
    std::string reconame = s.substr(0, s.find("_", s.find("sample")) ) + "_reconstruction";
    std::string fieldname = s.substr(0, s.find("_", s.find("sample")) ) + "_field";
    std::string name = "graph_"+s.substr(1+s.find("_",s.find("recoLA")));
    name.erase(name.find("_sample"),name.find("_",name.find("sample"))-name.find("_sample"));

    p.reco = book(reconame)->GetMean();
    p.recoErr = book(reconame)->GetRMS();
    p.entries = (unsigned)((*sample)->GetEntries());
    p.field = book(fieldname)->GetMean();
    harvest[name].push_back(p);
  }
  return harvest;
}

TGraphErrors* LA_Filler_Fitter::
tgraphe(const std::vector<Result>& vpoints) {
  std::vector<float> x,y,xerr,yerr;
  BOOST_FOREACH(const Result& p, vpoints) {
    if( p.chi2 / p.ndof < 10 ) {
      x.push_back(p.reco);
      y.push_back(p.measure);
      xerr.push_back(p.recoErr);
      yerr.push_back(p.measureErr);
    }
  }
  return new TGraphErrors(x.size(), &(x[0]),&(y[0]),&(xerr[0]),&(yerr[0]));
}

void LA_Filler_Fitter::
print_to_file(std::string filename, const std::vector<Result>& results) {
  fstream file(filename.c_str(),std::ios::out);
  file << "#reco\trecoErr\tmeasure\tmeasureErr\tchi2\tndof\tentries\tfield" << std::endl;
  BOOST_FOREACH(const Result& r, results) {
    file << r.reco       << "\t" 
	 << r.recoErr    << "\t"
	 << r.measure    << "\t"
	 << r.measureErr << "\t"
	 << r.chi2       << "\t"
	 << r.ndof       << "\t"
	 << r.entries    << "\t"
	 << r.field      << std::endl;
  }
  file.close();
}
