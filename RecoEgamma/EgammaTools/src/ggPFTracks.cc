#include "RecoEgamma/EgammaTools/interface/ggPFTracks.h"
//Class by Rishi Patel rpatel@cern.ch for Single Leg (high pt leg of Conversion
//For Vertex Selection, also has functions to do pointing using conversion pairs//that are stored with PFPhotons/PFElectrons

ggPFTracks::ggPFTracks(
		       edm::Handle<BeamSpot>& beamSpotHandle
		       ):
  beamSpotHandle_(beamSpotHandle),//pass beamspot
  isConv_(false)
{
  
}

ggPFTracks::~ggPFTracks(){}
//Fills a Vector of Conversion Tracks (single legs, or track pairs from a PFPhoton) and also Conversion Objects
void ggPFTracks::getPFConvTracks(
				 reco::Photon phot,
				 vector<edm::RefToBase<reco::Track> > &Tracks, 
				 reco::ConversionRefVector &conversions,
				 vector<edm::RefToBase<reco::Track> > &SLTracks, 
				 reco::ConversionRefVector &SLconversions
				 ){
  Tracks.clear();
  conversions.clear();
  SLTracks.clear();
  SLconversions.clear();
  conversions=phot.conversions();
  //loop over conversion paris
  for(unsigned int c=0; c<conversions.size(); ++c){
    const std::vector<edm::RefToBase<reco::Track> > tracks = conversions[c]->tracks();
    for (unsigned int t=0; t<tracks.size(); t++){
      Tracks.push_back(tracks[t]);
    }
  }
  
  reco::ConversionRefVector  SLConversions=phot.conversionsOneLeg();
  
  for(unsigned int SLc=0; SLc<SLConversions.size(); ++SLc){
    const std::vector<edm::RefToBase<reco::Track> > tracks = SLConversions[SLc]->tracks();
    SLconversions.push_back(SLConversions[SLc]);
    
    for (unsigned int t=0; t<tracks.size(); t++){
      SLTracks.push_back(tracks[t]);
    }
  }
  if(SLConversions.size()>0 || conversions.size()>0)isConv_=true;
}
//This function does SuperCluster Pointing using Single Legs and Conversion pairs
//There is a bool that is flagged when it uses SLconv exclusively
std::pair<float,float> ggPFTracks::BeamLineInt(	
					       reco::SuperClusterRef sc, 
					       vector<edm::RefToBase<reco::Track> > &Tracks, 
					       reco::ConversionRefVector &conversions,
					       vector<edm::RefToBase<reco::Track> > &SLTracks, 
					       reco::ConversionRefVector &SLconversions
					       ){
  std::pair<float,float> Zint(0,0);
  TVector3 beamspot(beamSpotHandle_->position().x(),beamSpotHandle_->position().y(),
	      beamSpotHandle_->position().z());
  TVector3 SCPos(sc->position().x()-beamspot[0], sc->position().y()-beamspot[1], sc->position().z()-beamspot[2]);
  //find min Conversion radius for track pairs if they exist. 
  float convRMin=130;
  float SLConvRMin=130;
  int c_index=-1; int SLc_index=-1;
  if(conversions.size()>0)
    {
      for(unsigned int c=0; c<conversions.size(); ++c){
	float convR=sqrt(conversions[c]->conversionVertex().x()* conversions[c]->conversionVertex().x() + conversions[c]->conversionVertex().y()* conversions[c]->conversionVertex().y());
	if(convRMin>convR){
	  convRMin=convR;
	  c_index=c;
	}
      }
    }
  
  if(SLconversions.size()>0){
    for(unsigned int SLc=0; SLc<SLconversions.size(); ++SLc){      
      std::vector<math::XYZPointF> innerPos=SLconversions[SLc]->tracksInnerPosition();
      for (unsigned int t=0; t<innerPos.size(); t++){
	float convR=sqrt( innerPos[t].X()* innerPos[t].X() + innerPos[t].Y()* innerPos[t].Y());
	
	if(SLConvRMin>convR){
	  SLc_index=SLc;  
	  SLConvRMin=convR;
	  
	}
      }
    }
  }
  TVector3 TkPos(beamspot[0],beamspot[1],beamspot[2]);
  //take the smaller Radius:
  if(convRMin<SLConvRMin && c_index>-1){
    //point using Conversion Vertex
    TkPos.SetXYZ(conversions[c_index]->conversionVertex().x()-beamspot[0],
		 conversions[c_index]->conversionVertex().y()-beamspot[1],
		 conversions[c_index]->conversionVertex().z()-beamspot[2] );
  }
  if(SLConvRMin<convRMin && SLc_index>-1){
    reco::Vertex conv=SLconversions[SLc_index]->conversionVertex();
    TkPos.SetXYZ(conv.x()-beamspot[0],
		 conv.y()-beamspot[1],
		 conv.z()-beamspot[2] );
  }
  //Intersection fromt the two points:
  float R1=sqrt(SCPos.X()* SCPos.X() + SCPos.Y()*SCPos.Y()); 
  float R2=sqrt(TkPos.X()* TkPos.X() + TkPos.Y()*TkPos.Y());
  float Z1=SCPos.Z();
  float Z2=TkPos.Z();
  float slope=(Z1-Z2)/(R1-R2);
  Zint.first=Z2 - R2*slope;
  //determine error based on Tracking Region for conversion pairs:
  float sigmaPix=0.06;
  float sigmaTib=0.67;
  float sigmaTob=2.04;
  float sigmaFwd1=0.18;
  float sigmaFwd2=0.61;
  float sigmaFwd3=0.99;
  //error of SL tracks of Conversion based on EB/EE and R>39 R<39 (4 cat)
  float EBLR=0.24;
  float EBHR=0.478;
  float EELR=0.416;
  float EEHR=0.888;
  if(sc->eta()<1.4442){//Barrel
    //if conversion
    if(convRMin<SLConvRMin && c_index>-1){//3 tracking regions
      if(convRMin<=15)Zint.second=sigmaPix;
      else if(convRMin>15 && convRMin<=60)Zint.second=sigmaTib;
      else Zint.second=sigmaTob;
    }  
    //if SL 
    if(SLConvRMin<convRMin && SLc_index>-1){//2 tracking regions
      if(SLConvRMin<39)Zint.second=EBLR;
      else Zint.second=EBHR;
    }
  }
  else{
    //if conversion
    if(convRMin<SLConvRMin && c_index>-1){//3 foreward tracking regions
      float convz=conversions[c_index]->conversionVertex().z();
      if(convz<=50)Zint.second=sigmaFwd1;
      else if(convz>50 && convz<=60)Zint.second=sigmaFwd2;
      else Zint.second=sigmaFwd3;
    }
    //if SL
    if(SLConvRMin<convRMin && SLc_index>-1){//2 tracking regions based on R
      if(SLConvRMin<39)Zint.second=EELR;
      else Zint.second=EEHR;
    }
  }
  return Zint;//return intersection at beamline and error in the pointing based on tracking region
}
//This Function does the track Projection (using innermost hit and inner momentum of the Single Track, NOTE: Can also use for GSF tracks
std::pair<float,float> ggPFTracks::TrackProj(
					     bool isEb, 
					     reco::GsfTrackRef gsf,
					     vector<edm::RefToBase<reco::Track> > &SLTracks, 
					     reco::ConversionRefVector &SLconversions
			    ){
  std::pair<float,float> ZProj(0,0);
  
  if(gsf.isNonnull()){//if there is a gsf track then use this for track projection Plenty of inner hits
    
    float theta =gsf->innerMomentum().theta();
    
    float tkz=gsf->innerPosition().Z();
    float tkR=sqrt(gsf->innerPosition().X()* gsf->innerPosition().X()+ gsf->innerPosition().Y()* gsf->innerPosition().Y());
    float thetErr=gsf->thetaError();
    float Z=tkz-tkR/tan(theta);
    float Zerr=((-1*(cos(theta)*cos(theta))/(sin(theta)* sin(theta))-1)*tkR*thetErr);
    ZProj.first=Z; ZProj.second=Zerr;
    return ZProj;
  }
  
  float minR=210;
  int SLind=-1;
  if(SLconversions.size()>0){//find track with min Radius (starts earliest)
    for(unsigned int SLc=0; SLc<SLconversions.size(); ++SLc){     
      reco::Vertex conv=SLconversions[SLc]->conversionVertex();
      float convR=sqrt( conv.x() * conv.x() + conv.y() * conv.y());
      if(convR<minR){
	minR=convR;
	SLind=SLc;
      }
    }
    reco::Vertex conv=SLconversions[SLind]->conversionVertex();
    const std::vector<edm::RefToBase<reco::Track> > tracks = SLconversions[SLind]->tracks();
    float theta =tracks[0]->theta();
    float tkz=conv.z();
    float tkR=sqrt( conv.x() * conv.x() + conv.y() * conv.y());
    float thetErr=tracks[0]->thetaError();
    float Z=tkz-tkR/tan(theta);//track projection
    float Zerr=((-1*(cos(theta)*cos(theta))/(sin(theta)* sin(theta))-1)*tkR*thetErr); //projection error based on theta err of track theta derivative of the Zproj
    //for early tracks theta error is very small so just hard code an error 
    //that is measured by looking at the track Proj resolutin in MC
    if(tkR<39 && isEb)Zerr=0.234; 
    if(tkR<39 && !isEb)Zerr=0.341;
    ZProj.first=Z; ZProj.second=Zerr;
    return ZProj;
  }
  return ZProj;
}

//this function combines the results from the Track Projection and SC pointing 
//can also use conversion pairs and gsf tracks, even when there is no Single leg
std::pair<float,float> ggPFTracks::CombZVtx(
			   reco::SuperClusterRef sc, 
			   reco::GsfTrackRef gsf,
			   vector<edm::RefToBase<reco::Track> > &Tracks, 
			   reco::ConversionRefVector &conversions,
			   vector<edm::RefToBase<reco::Track> > &SLTracks, 
			   reco::ConversionRefVector &SLconversions
			   
			   ){
  std::pair<float, float> combZ(0,0);
  bool isEb;
  TVector3 beamspot(beamSpotHandle_->position().x(),beamSpotHandle_->position().y(),
		    beamSpotHandle_->position().z());
  if(fabs(sc->eta())<1.4442)isEb=true;
  else isEb=false;
  std::pair< float,float> SCZ=BeamLineInt(sc, Tracks, conversions,SLTracks, SLconversions);
  std::pair<float, float> TkPjZ=TrackProj(isEb,gsf,SLTracks, SLconversions);
  //errors in the two methods
  float sigZProj=TkPjZ.second;
  float sigSCPoint=SCZ.second;
  
  if(gsf.isNonnull()){combZ=TkPjZ; return combZ;}//for gsf Tracks just return track Proj
  //weighted avg of the two methods, where weights are based on the error
  float Z=((SCZ.first/(sigSCPoint*sigSCPoint))+ (TkPjZ.first/( sigZProj* sigZProj)))/(1/(sigSCPoint * sigSCPoint)+ 1/(sigZProj * sigZProj));
  //total error sum of the two errors in quadrature
  float sigZ=sqrt((sigSCPoint * sigSCPoint)+ (sigZProj * sigZProj));
  combZ.first=Z; combZ.second=sigZ;
  return combZ;  
}
//this function combines the results from the Track Projection and SC pointing 
//simpler version of the above function, but returns bspot Z when there is no SL
std::pair<float, float> ggPFTracks::SLCombZVtx(
					       reco::Photon phot,
					       bool &hasSL
					       ){
  std::pair<float, float> combZ(0,0);
  TVector3 beamspot(beamSpotHandle_->position().x(),beamSpotHandle_->position().y(),
		    beamSpotHandle_->position().z());
  bool isEb=phot.isEB();
  //only want to use SL so the conversion pair variables will be dummy for the get function
  vector<edm::RefToBase<reco::Track> >convTracks; 
  reco::ConversionRefVector pairConv;
  
  reco::ConversionRefVector SLConversions;
  vector<edm::RefToBase<reco::Track> >SLTracks; 
  reco::GsfTrackRef gsfdummy;
  getPFConvTracks(phot,convTracks,pairConv ,SLTracks,SLConversions);
  
  if(SLConversions.size()>0){
    hasSL=true;
    vector<edm::RefToBase<reco::Track> >dummy; 
    reco::ConversionRefVector pairdummy;
    std::pair<float, float> TkPjZ=TrackProj(isEb,gsfdummy,SLTracks, SLConversions);
    std::pair< float,float> SCZ=BeamLineInt(phot.superCluster(), dummy,pairdummy,SLTracks, SLConversions);
    //errors in the two methods
    float sigZProj=TkPjZ.second;
    float sigSCPoint=SCZ.second;
    //weighted avg of the two methods, where weights are based on the error
    float Z=((SCZ.first/(sigSCPoint*sigSCPoint))+ (TkPjZ.first/(sigZProj * sigZProj)))/(1/(sigSCPoint*sigSCPoint)+ 1/(sigZProj * sigZProj));
    //total error sum of the two errors in quadrature
    float sigZ=sqrt((sigSCPoint*sigSCPoint)+ (sigZProj * sigZProj));
    combZ.first=Z; combZ.second=sigZ;
  }
  else if(pairConv.size()>0){//else use Conversion pairs if available 
    std::pair< float,float> SCZ=BeamLineInt(phot.superCluster(),convTracks, pairConv, SLTracks, SLConversions);   
    combZ.first=SCZ.first;
    combZ.second=SCZ.second;
    hasSL=false;
  }
  else{//returns beamspot
    combZ.first=beamspot.Z();
    combZ.second=0;
    hasSL=false;
  }
  return combZ;
}
