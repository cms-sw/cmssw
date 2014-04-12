///////////////////////////////////////////////////////////////////////////////
// File: ClusterProducerFP420.cc
// Date: 12.2006
// Description: ClusterProducerFP420 for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
#include "RecoRomanPot/RecoFP420/interface/ClusterProducerFP420.h"
using namespace std;

// sense of xytype here is X or Y type planes. Now we are working with X only, i.e. xytype=2

bool ClusterProducerFP420::badChannel( int channel, 
				       const std::vector<short>& badChannels) const
{
  const std::vector<short>::size_type linearCutoff = 20;// number of possible bad channels
  // check: is it bad cnannel or not
    /*
  std::cout  
    << "badChannel:  badChannels.size()= " << badChannels.size() <<  " \t"
    << "badChannel:  hardcoded linearCutoff= " << linearCutoff  << " \t"
    << "badChannel:  channel= " << channel <<  " \t"
    << std::endl; 
*/
  if (badChannels.size() < linearCutoff) {
    return (std::find( badChannels.begin(), badChannels.end(), channel) != badChannels.end());
  }
  else return std::binary_search( badChannels.begin(), badChannels.end(), channel);


}


//FIXME
//In the future, with blobs, perhaps we will come back at this version
// std::vector<ClusterFP420>
// ClusterProducerFP420::clusterizeDetUnit( DigiIterator begin, DigiIterator end,
//                                                 unsigned int detid,
//                                                 const std::vector<float>& noiseVec,
//                                                 const std::vector<short>& badChannels)
// {

//                                                                              digiRangeIteratorBegin,digiRangeIteratorEnd
std::vector<ClusterFP420> ClusterProducerFP420::clusterizeDetUnit( HDigiFP420Iter begin, HDigiFP420Iter end,
								   unsigned int detid, const ElectrodNoiseVector& vnoise){
//                                                 const std::vector<short>& badChannels)

  //reminder:	  int zScale=2;  unsigned int detID = sScale*(sector - 1)+zScale*(zmodule - 1)+xytype;
  // const int maxBadChannels_ = 1;
  HDigiFP420Iter ibeg, iend, ihigh, itest, i;  
  ibeg = iend = begin;
  std::vector<HDigiFP420> cluster_digis;  
  // reserve 15 possible channels for one cluster
  cluster_digis.reserve(15);
  // reserve one third of digiRange for number of clusters
  std::vector<ClusterFP420> rhits; rhits.reserve( (end - begin)/3 + 1);
  //  predicate(declare): take noise above seed_thr
  AboveSeed predicate(seedThresholdInNoiseSigma(),vnoise);
  //Check if channel is lower than vnoise.size()
  itest = end - 1;
  int vnoisesize = vnoise.size();
  //  if (vnoise.size()<=itest->channel()) // old
  if (vnoisesize<=itest->channel()) 
    {
      std::cout <<  "WARNING for detid " << detid << " there will be a request for noise for channel seed" << itest->channel() << " but this detid has vnoise.size= " <<  vnoise.size() << "\nskip"<< std::endl;
      return rhits;
    }
  //
  //                                   loop in elements above seed_thr
  //                                find seed with seed noise above seed_thr
  while ( ibeg != end && (ihigh = find_if( ibeg, end, predicate)) != end) {
    // The seed electrode is ihigh. Scan up and down from it, finding nearby(sosednie) electrodes above
    // threshold, allowing for some voids. The accepted cluster runs from electrode ibeg
    // to iend, and itest is the electrode under study, not yet accepted.
    
    // go to right side:
    iend = ihigh;
    itest = iend + 1;
    while ( itest != end && (itest->channel() - iend->channel() <= max_voids_ + 1 )) {
      float channelNoise = vnoise[itest->channel()].getNoise();
      bool IsBadChannel = vnoise[itest->channel()].getDisable();
      if (!IsBadChannel && itest->adc() >= static_cast<int>( channelThresholdInNoiseSigma() * channelNoise)) { 
	iend = itest;
      }
      ++itest;
    }
    //if the next digi after iend is an adjacent bad(!) digi then insert into candidate cluster
    itest=iend+1;
    if ( itest != end && (itest->channel() - iend->channel() == 1) && vnoise[itest->channel()].getDisable() ) {    
      std::cout << "Inserted bad electrode at the end edge iend->channel()= " << iend->channel() << " itest->channel() = " << itest->channel() << std::endl;
      iend++;
    }
    // go to left side:
    ibeg = ihigh;
    itest = ibeg - 1;
    while ( itest >= begin && (ibeg->channel() - itest->channel() <= max_voids_ + 1 )) {
      float channelNoise = vnoise[itest->channel()].getNoise();  
      bool IsBadChannel = vnoise[itest->channel()].getDisable();
      if (!IsBadChannel && itest->adc() >= static_cast<int>( channelThresholdInNoiseSigma() * channelNoise)) { 
        ibeg = itest;
      }
      --itest;
    }
    //if the next digi after ibeg is an adiacent bad digi then insert into candidate cluster
    itest=ibeg-1;
    if ( itest >= begin && (ibeg->channel() - itest->channel() == 1) && vnoise[itest->channel()].getDisable() ) {    
      std::cout << "Inserted bad electrode at the begin edge ibeg->channel()= " << ibeg->channel() << " itest->channel() = " << itest->channel() << std::endl;
      ibeg--;
    }
    //============================================================================================================
    int charge = 0;
    float sigmaNoise2=0;
    cluster_digis.clear();
    for (i=ibeg; i<=iend; ++i) {
      float channelNoise = vnoise[i->channel()].getNoise();  
      bool IsBadChannel = vnoise[i->channel()].getDisable();
      //just check for consecutive digis
      if (i!=ibeg && i->channel()-(i-1)->channel()!=1){
	//digits: *(i-1) and *i   are not consecutive(we asked !=1-> it means 2...),so  create an equivalent number of Digis with zero amp
	for (int j=(i-1)->channel()+1;j<i->channel();++j){
	  cluster_digis.push_back(HDigiFP420(j,0)); //if electrode bad or under threshold set HDigiFP420.adc_=0  
	}
      }
      //

// FIXME: should the digi be tested for badChannel before using the adc?

      if (!IsBadChannel && i->adc() >= static_cast<int>( channelThresholdInNoiseSigma()*channelNoise)) {
        charge += i->adc();
        sigmaNoise2 += channelNoise*channelNoise; // 
        cluster_digis.push_back(*i);// put into cluster_digis good i info
      } else {
	cluster_digis.push_back(HDigiFP420(i->channel(),0)); //if electrode bad or under threshold set HDigiFP420.adc_=0
      }
      //
    }//for i++
    float sigmaNoise = sqrt(sigmaNoise2);
    // define here cog,err,xytype not used
    float cog;
    float err;
    unsigned int xytype=2;// it can be even =1,although we are working with =2(Xtypes of planes)
    if (charge >= static_cast<int>( clusterThresholdInNoiseSigma()*sigmaNoise)) {
      rhits.push_back( ClusterFP420( detid, xytype, ClusterFP420::HDigiFP420Range( cluster_digis.begin(),
									   cluster_digis.end()), 
				     cog, err));
      //      std::cout << "Looking at cog and err  : cog " << cog << " err " << err  << std::endl;
    }
    ibeg = iend+1;
  } // while ( ibeg
  return rhits;
}

int ClusterProducerFP420::difNarr(unsigned int xytype, HDigiFP420Iter ichannel,
				  HDigiFP420Iter jchannel) {
  int d = 9999;
    if(xytype == 2) {
      d = ichannel->stripV() - jchannel->stripV();
      d=std::abs(d);
    }
    else if(xytype == 1) {
      d = ichannel->stripH() - jchannel->stripH();
      d=std::abs(d);
    }
    else{
      std::cout << "difNarr: wrong xytype =  " << xytype << std::endl;
    }
  return d;
}
int ClusterProducerFP420::difWide(unsigned int xytype, HDigiFP420Iter ichannel,
				  HDigiFP420Iter jchannel) {
  int d = 9999;
    if(xytype == 2) {
      d = ichannel->stripVW() - jchannel->stripVW();
      d=std::abs(d);
    }
    else if(xytype == 1) {
      d = ichannel->stripHW() - jchannel->stripHW();
      d=std::abs(d);
    }
    else{
      std::cout << "difWide: wrong xytype =  " << xytype << std::endl;
    }
  return d;
}
//                                                                              digiRangeIteratorBegin,digiRangeIteratorEnd
std::vector<ClusterFP420> ClusterProducerFP420::clusterizeDetUnitPixels( HDigiFP420Iter begin, HDigiFP420Iter end,
									 unsigned int detid, const ElectrodNoiseVector& vnoise, unsigned int xytype, int verb){
//                                                 const std::vector<short>& badChannels)

  //reminder:	  int zScale=2;  unsigned int detID = sScale*(sector - 1)+zScale*(zmodule - 1)+xytype;

  // const int maxBadChannels_ = 1;
  
  HDigiFP420Iter ibeg, iend, ihigh, itest, i;  
  ibeg = iend = begin;
  std::vector<HDigiFP420> cluster_digis;  
  
  // reserve 25 possible channels for one cluster
  cluster_digis.reserve(25);
  
  // reserve one third of digiRange for number of clusters
  std::vector<ClusterFP420> rhits; rhits.reserve( (end - begin)/3 + 1);
  
  //  predicate(declare): take noise above seed_thr
  AboveSeed predicate(seedThresholdInNoiseSigma(),vnoise);
  
  //Check if no channels with digis at all
  /*
  HDigiFP420Iter abeg, aend;  
  abeg = begin; aend = end;
  std::vector<HDigiFP420> a_digis;  
  for ( ;abeg != aend; ++abeg ) {
    a_digis.push_back(*abeg);
  } // for
  if (a_digis.size()<1) return rhits;;
*/  
  //Check if channel is lower than vnoise.size()
  itest = end - 1;
  int vnoisesize = vnoise.size();
  if (vnoisesize<=itest->channel()) 
    {
//      std::cout <<  "WARNING for detid " << detid << " there will be a request for noise for channel seed" << itest->channel() << " but this detid has vnoise.size= " <<  vnoise.size() << "\nskip"<< std::endl;
      return rhits;
    }
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  
  //  std::cout << "before while loop..." << std::endl;
  
  //                                   loop in elements above seed_thr
  //                                find seed with seed noise above seed_thr
  while ( ibeg != end && (ihigh = find_if( ibeg, end, predicate)) != end) {
    
    
    // The seed electrode is ihigh. Scan up and down from it, finding nearby(sosednie) electrodes above
    // threshold, allowing for some voids. The accepted cluster runs from electrode ibeg
    // to iend, and itest is the electrode under study, not yet accepted.
    
    // go to right side:
    iend = ihigh;
    itest = iend + 1;
    //    while ( itest != end && (itest->channel() - iend->channel() <= max_voids_ + 1 )) {
    while ( itest != end && (difNarr(xytype,itest,iend)<= max_voids_ + 1 ) && (difWide(xytype,itest,iend)<= 1) ) {
      float channelNoise = vnoise[itest->channel()].getNoise();
      bool IsBadChannel = vnoise[itest->channel()].getDisable();
      if (!IsBadChannel && itest->adc() >= static_cast<int>( channelThresholdInNoiseSigma() * channelNoise)) { 
	iend = itest;
	if(verb>2){
	  std::cout << "=========================================================================== " << std::endl;
	  std::cout << "Right side: itest->adc()= " << itest->adc() << " channel_noise = " << static_cast<int>( channelThresholdInNoiseSigma() * channelNoise) << std::endl;
	}
      }
      ++itest;
    }
    //if the next digi after iend is an adjacent bad(!) digi then insert into candidate cluster
    itest=iend+1;
    if ( itest != end && (difNarr(xytype,itest,iend) == 1) && (difWide(xytype,itest,iend)< 1) && vnoise[itest->channel()].getDisable() ) {    
      if(verb>2){
	std::cout << "Inserted bad electrode at the end edge iend->channel()= " << iend->channel() << " itest->channel() = " << itest->channel() << std::endl;
      }
      iend++;
    }
    if(verb>2){
      std::cout << "Result of going to right side iend->channel()= " << iend->channel() << " itest->channel() = " << itest->channel() << std::endl;
    }
    
    // go to left side:
    ibeg = ihigh;
    itest = ibeg - 1;
    //  while ( itest >= begin && (ibeg->channel() - itest->channel() <= max_voids_ + 1 )) {
    while ( itest >= begin && (difNarr(xytype,ibeg,itest) <= max_voids_ + 1 ) && (difWide(xytype,ibeg,itest) <= 1) ) {
      float channelNoise = vnoise[itest->channel()].getNoise();  
      bool IsBadChannel = vnoise[itest->channel()].getDisable();
      if (!IsBadChannel && itest->adc() >= static_cast<int>( channelThresholdInNoiseSigma() * channelNoise)) { 
        ibeg = itest;
	if(verb>2){
	  std::cout << "Left side: itest->adc()= " << itest->adc() << " channel_noise = " << static_cast<int>( channelThresholdInNoiseSigma() * channelNoise) << std::endl;
	}
      }
      --itest;
    }
    //if the next digi after ibeg is an adjacent bad digi then insert into candidate cluster
    itest=ibeg-1;
    if ( itest >= begin && (difNarr(xytype,ibeg,itest) == 1) && (difWide(xytype,ibeg,itest) <  1) && vnoise[itest->channel()].getDisable() ) {    
      if(verb>2){
	std::cout << "Inserted bad electrode at the begin edge ibeg->channel()= " << ibeg->channel() << " itest->channel() = " << itest->channel() << std::endl;
      }
      ibeg--;
    }
    if(verb>2){
      std::cout << "Result of going to left side ibeg->channel()= " << ibeg->channel() << " itest->channel() = " << itest->channel() << std::endl;
    }
    //============================================================================================================
    
    
    
    
    
    //============================================================================================================
    int charge = 0;
    float sigmaNoise2=0;
    cluster_digis.clear();
    //    HDigiFP420Iter ilast=ibeg; // AZ
    if(verb>2){
      std::cout << "check for consecutive digis ibeg->channel()= " << ibeg->channel() << " iend->channel() = " << iend->channel() << std::endl;
    }
    for (i=ibeg; i<=iend; ++i) {
      float channelNoise = vnoise[i->channel()].getNoise();  
      bool IsBadChannel = vnoise[i->channel()].getDisable();
      if(verb>2){
	std::cout << "Looking at cluster digis: detid " << detid << " digis " << i->channel()  
		  << " adc " << i->adc() << " channelNoise " << channelNoise << " IsBadChannel  " << IsBadChannel << std::endl;
      }
      
      //just check for consecutive digis
      // if (i!=ibeg && i->channel()-(i-1)->channel()!=1){
      //if (i!=ibeg && difNarr(xytype,i,i-1) !=1 && difWide(xytype,i,i-1) !=1){
      if(verb>2){
	std::cout << "difNarr(xytype,i,i-1) = " << difNarr(xytype,i,i-1)  << std::endl;
	std::cout << "difWide(xytype,i,i-1) = " << difWide(xytype,i,i-1)  << std::endl;
      }
      // in fact, no sense in this check, but still keep if something wrong is going:
      //      if (i!=ibeg && (difNarr(xytype,i,i-1) > 1 || difWide(xytype,i,i-1) > 1)   ){
      if (i!=ibeg && (difNarr(xytype,i,i-1) > 1 && difWide(xytype,i,i-1) > 1)   ){
	//digits: *(i-1) and *i   are not consecutive(we asked !=1-> it means 2...),so  create an equivalent number of Digis with zero amp
	for (int j=(i-1)->channel()+1;j<i->channel();++j){
	  if(verb>2){
	    std::cout << "not consecutive digis: set HDigiFP420.adc_=0 : j = " << j  << std::endl;
	  }
	  cluster_digis.push_back(HDigiFP420(j,0)); //if not consecutive digis set HDigiFP420.adc_=0  
	}//for
      }//if
      
      
      if (!IsBadChannel && i->adc() >= static_cast<int>( channelThresholdInNoiseSigma()*channelNoise)) {
        charge += i->adc();
        sigmaNoise2 += channelNoise*channelNoise; // 
        cluster_digis.push_back(*i);// put into cluster_digis good i info
	if(verb>2){
	  std::cout << "put into cluster_digis good i info: i->channel() = " << i->channel()  << std::endl;
	}
      } else {
	cluster_digis.push_back(HDigiFP420(i->channel(),0)); //if electrode bad or under threshold set HDigiFP420.adc_=0
	if(verb>2){
	  std::cout << "else if electrode bad or under threshold set HDigiFP420.adc_=0: i->channel() = " << i->channel()  << std::endl;
	}
      }//if else
      
    }//for i++
    
    
    
    
    float sigmaNoise = sqrt(sigmaNoise2);
    float cog;
    float err;
    if (charge >= static_cast<int>( clusterThresholdInNoiseSigma()*sigmaNoise)) {
      rhits.push_back( ClusterFP420( detid, xytype, ClusterFP420::HDigiFP420Range( cluster_digis.begin(),
										  cluster_digis.end()), 
				     cog, err));
      if(verb>2){
	std::cout << "Looking at cog and err  : cog " << cog << " err " << err  << std::endl;
	std::cout << "=========================================================================== " << std::endl;
      }
    }
    
    
    ibeg = iend+1;
  } // while ( ibeg
  
  
  return rhits;
  
}



