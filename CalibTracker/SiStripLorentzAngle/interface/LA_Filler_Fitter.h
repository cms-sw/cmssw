#ifndef LA_FILLER_FITTER_H
#define LA_FILLER_FITTER_H

#include <cmath>
#include <string>
#include <vector>
#include <map>
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
  static void fit(Book& book) {make_and_fit_ratio(book); make_and_fit_profile(book,"-tanLA");}
  static void make_and_fit_ratio(Book&);
  static void make_and_fit_profile(Book&, const std::string&);
  static std::map<std::string, TGraphErrors*> harvest_samples(const Book&);

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

  struct Point { float x,y,xerr,yerr; Point(){}};
  static TGraphErrors* tgraphe(const std::vector<Point>& vpoints);

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
    book(ratio_name)->Fit("gaus");
    double mean = book(ratio_name)->GetFunction("gaus")->GetParameter(1);
    double sigma = book(ratio_name)->GetFunction("gaus")->GetParameter(2);
    book(ratio_name)->Fit("gaus","","",mean-sigma,mean+sigma);
  }
}

void LA_Filler_Fitter::
make_and_fit_profile(Book& book, const std::string& key) {
  for(Book::const_iterator hist2D = book.begin(".*"+key); hist2D!=book.end(); ++hist2D) {
    std::string name = hist2D.name()+"_profile";
    book.book(name, (TH1*) ((TH2*)(*hist2D))->ProfileX(name.c_str()));
    TF1* fit = new TF1("fitfunc","[1]*(TMath::Abs(x-[0]))+[2]",-1,1);
    fit->SetParameters(book(name)->GetMean(1),
		       -0.5 * book(name)->GetMaximum() / book(name)->GetBinCenter(0),
		       0.5*book(name)->GetMean(2));
    book(name)->Fit(fit,"IMQ");
  }
}

std::map<std::string, TGraphErrors*> LA_Filler_Fitter::
harvest_samples(const Book& book) {
  typedef std::map<std::string, std::vector<Point> > harvest_t;

  harvest_t harvest;
  for(Book::const_iterator sample = book.begin(".*_sample.*"); sample!=book.end(); ++sample ) {

    Point p;

    if( sample.name().find("ratio") != std::string::npos ) {
      p.y = (*sample)->GetFunction("gaus")->GetParameter(1);
      p.yerr = (*sample)->GetFunction("gaus")->GetParError(1);
    } else 
    if( sample.name().find("profile") != std::string::npos ) {
      p.y = (*sample)->GetFunction("fitfunc")->GetParameter(0);
      p.yerr = (*sample)->GetFunction("fitfunc")->GetParError(0);
    } else continue;

    std::string s = sample.name();
    std::string reconame = s.substr(0, s.find("_", s.find("sample")) ) + "_reconstruction";
    std::string name = "graph_"+s.substr(1+s.find("_",s.find("recoLA")));
    name.erase(name.find("_sample"),name.find("_",name.find("sample"))-name.find("_sample"));

    p.x = book(reconame)->GetMean();
    p.xerr = book(reconame)->GetRMS();
    harvest[name].push_back(p);
  }
  
  std::map<std::string, TGraphErrors*> harvest_graphs;
  for(harvest_t::const_iterator it = harvest.begin(); it!=harvest.end(); ++it) {
    std::cout << it->first << std::endl;
    harvest_graphs[it->first] = tgraphe(it->second);
  }

  return harvest_graphs;
}

TGraphErrors* LA_Filler_Fitter::
tgraphe(const std::vector<Point>& vpoints) {
  std::vector<float> x,y,xerr,yerr;
  BOOST_FOREACH(Point p, vpoints) {
    x.push_back(p.x);
    y.push_back(p.y);
    xerr.push_back(p.xerr);
    yerr.push_back(p.yerr);
    std::cout << p.x << "(" << p.xerr << ")\t" << p.y << "(" << p.yerr << ")" << std::endl;
  }
  return new TGraphErrors(vpoints.size(), &(x[0]),&(y[0]),&(xerr[0]),&(yerr[0]));
}
