#ifndef LA_FILLER_FITTER_H
#define LA_FILLER_FITTER_H

#include <cmath>
#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <TTree.h>
#include "TF1.h"
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
  static void collate_ensemble(const Book&, Book&);

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
    fit->SetParameters(book(name)->GetMean(1),book(name)->GetMean(2),1);
    book(name)->Fit(fit);
  }
}

void LA_Filler_Fitter::
collate_ensemble(const Book& laBook, Book& book) {
  for(Book::const_iterator recoSample = laBook.begin(".*_sample\\d*_reconstruction"); recoSample!=laBook.end(); ++recoSample ) {
    std::string name = recoSample.name();
    std::string base = name.substr(0, name.find("_sample"));
    std::string sample = name.substr(name.find("_sample"),name.find("_reconstruction")-name.find("_sample"));

    TH1* ratio = laBook(base+sample+"_ratio_tanLA");
    TH1* wprofile = laBook(base+sample+"_width-tanLA_profile");
    TH1* vprofile = laBook(base+sample+"_variance-tanLA_profile");
    double true_value = (*recoSample)->GetMean();
    double ratio_value = ratio->GetFunction("gaus")->GetParameter(1);
    double ratio_pull = (ratio_value-true_value) / ratio->GetFunction("gaus")->GetParError(1);
    double wprofile_value = wprofile->GetFunction("fitfunc")->GetParameter(0);
    double wprofile_pull = (wprofile_value-true_value) / wprofile->GetFunction("fitfunc")->GetParError(0);
    double vprofile_value = vprofile->GetFunction("fitfunc")->GetParameter(0);
    double vprofile_pull = (vprofile_value-true_value) / vprofile->GetFunction("fitfunc")->GetParError(0);

    book.fill( ratio_value,    base+"_ratio_value",    101, -0.2, 0.1 );
    book.fill( wprofile_value, base+"_wprofile_value", 101, -0.2, 0.1 );
    book.fill( vprofile_value, base+"_vprofile_value", 101, -0.2, 0.1 );

    book.fill( ratio_pull,     base+"_ratio_pull",     101, -5, 5 );
    book.fill( wprofile_pull,  base+"_wprofile_pull",  101, -5, 5 );
    book.fill( vprofile_pull,  base+"_vprofile_pull",  101, -5, 5 );
  }
}
