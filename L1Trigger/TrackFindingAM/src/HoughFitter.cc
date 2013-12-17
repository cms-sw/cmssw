#include "../interface/HoughFitter.h"

HoughFitter::HoughFitter():TrackFitter(){
}

HoughFitter::HoughFitter(int nb):TrackFitter(nb){
}

HoughFitter::~HoughFitter(){
}

void HoughFitter::initialize(){

}

void HoughFitter::mergePatterns(){
  //cout<<"Merging of patterns not implemented"<<endl;
}

void HoughFitter::mergeTracks(){
  //cout<<"Merging of Tracks not implemented"<<endl;
}

void HoughFitter::fit(){

  int min_nb_stubs = 4;

  vector<Hit*> activatedHits;

  //////// Get the list of unique stubs from all the patterns ///////////
  set<int> ids;
  int total=0;
  for(unsigned int i=0;i<patterns.size();i++){
    vector<Hit*> allHits = patterns[i]->getHits();
    total+=allHits.size();
    for(unsigned int j=0;j<allHits.size();j++){
      pair<set<int>::iterator,bool> result = ids.insert(allHits[j]->getID());
      if(result.second==true)
	activatedHits.push_back(allHits[j]);
    }
  }
  ///////////////////////////////////////////////////////////////////////

  //low resolution hough
  HoughLocal* htl = new HoughLocal(0.5-PI/2,0,-0.01,0.01,32,32);

  //Add stubs to hough space
  //cout<<activatedHits.size()<<" selected stubs"<<endl;
  for(unsigned int j=0;j<activatedHits.size();j++){
    //cout<<j<<" : "<<*(activatedHits[j])<<endl;
    htl->fill(activatedHits[j]->getX(), activatedHits[j]->getY());
  }
  //cout<<endl;
  
  /*
    //DISPLAY HOUGH SPACE
  for(int i=0;i<32;i++){
    for(int j=0;j<32;j++){
      cout<<htl->getValue(i,j);
    }
    cout<<endl;
  }
  */
  std::vector < std::pair<double,double> > hbins;
  htl->findMaximumBins(hbins,min_nb_stubs-1);//we are more permissive the first time
  if (htl->getVoteMax()>=(unsigned int)min_nb_stubs-1){
    //cout<<"etape 1 OK (vote max = "<<htl->getVoteMax()<<" bins : "<<hbins.size()<<")"<<endl;
    for (std::vector < std::pair<double,double> >::iterator ihb=hbins.begin();ihb<hbins.end();ihb++){
      double ith=(*ihb).first;
      double ir=(*ihb).second;
				
      //printf("Bin  studied %f %f %f %f => \n",ith-2*htl->getThetaBin(),ith+2*htl->getThetaBin(),ir-2*htl->getRBin(),ir+2*htl->getRBin());
      //HoughLocal::PrintConvert(ith,ir);
      double R=1./2./TMath::Abs(ir);
      double pth=0.3*3.8*R/100.;
      uint32_t nbinf=64;
      if (pth<5) nbinf=128;
      if (pth>=5 && pth<10) nbinf=192;
      if (pth>=10  && pth<30) nbinf=256;
      if (pth>=30) nbinf=320;
      
      //for each candidate at low res, creates a new houh at high res
      HoughLocal *htp = new HoughLocal(ith-2*htl->getThetaBin(),ith+2*htl->getThetaBin(),ir-2*htl->getRBin(),ir+2*htl->getRBin(),nbinf,nbinf);
      
      htp->clear();

      //add the stubs
      for(unsigned int j=0;j<activatedHits.size();j++){
	htp->fill(activatedHits[j]->getX(), activatedHits[j]->getY());
      }
      /*
      for(unsigned int i=0;i<nbinf;i++){
	for(unsigned int j=0;j<nbinf;j++){
	  cout<<htp->getValue(i,j);
	}
	cout<<endl;
      }
      */
      std::vector< std::pair<double,double> > hfbins;hfbins.clear();
	
      //cout<<"etape 2 (vote max = "<<htp->getVoteMax()<<")"<<endl;
    
      //we are more restrictive at high res
      if (htp->getVoteMax()<(unsigned int)min_nb_stubs){
	delete htp;
	continue;
      }

      //we will keep only one candidate at high res : let's keep only the best ones
      htp->findMaximumBins(hfbins,htp->getVoteMax());//we know getVoteMax is at least min_nb_stubs
      
      //cout<<"Bins : "<<hfbins.size()<<endl;

      if(hfbins.size()>0){
	for (std::vector < std::pair<double,double> >::iterator ihbp=hfbins.begin();ihbp<hfbins.end();ihbp++){
	  double theta=(*ihbp).first;
	  double r=(*ihbp).second;
	  
	  double a=-1./tan(theta);

	  double R=1./2./TMath::Abs(r);
	  double pt=0.3*3.8*R/100.;
	  double phi=atan(a);
	  if(pt<1.5){
	    //cout<<"PT too small -> delete track"<<endl;
	    continue;
	  }
	  if(pt>1000){
	    //cout<<"PT over 1TeV -> delete track"<<endl;
	    continue;
	  }

	  //cout<<"Track found : "<<pt<<" GeV/c Phi="<<phi<<endl;
	  Track* fit_track = new Track(pt, 0, phi, 0, 0);
	  tracks.push_back(fit_track);
	  //cout<<"skip other bins"<<endl;
	  break;
	}
      }

      delete(htp);

    }
  }
  else{
    //cout<<"etape 1 FAILED (vote max = "<<htl->getVoteMax()<<")"<<endl;
  }

  delete(htl);
}

TrackFitter* HoughFitter::clone(){
  HoughFitter* fit = new HoughFitter(nb_layers);
  fit->setPhiRotation(sec_phi);
  return fit;
}
