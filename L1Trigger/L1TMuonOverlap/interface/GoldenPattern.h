#ifndef OMTF_GoldenPattern_H
#define OMTF_GoldenPattern_H

#include <vector>
#include <ostream>

#include "L1Trigger/L1TMuonOverlap/interface/OMTFinput.h"

class OMTFConfigMaker;
class OMTFProcessor;
class OMTFConfiguration;
//////////////////////////////////
// Key
//////////////////////////////////
struct Key {

Key(int iEta=99, unsigned int iPt=0, int iCharge= 0): 
  theEtaCode(iEta), thePtCode(iPt), theCharge(iCharge){}
    
    inline bool operator< (const Key & o) const {
    if (thePtCode > o.thePtCode) return true;
    else if (thePtCode==o.thePtCode && theCharge < o.theCharge) return true;
    else if (theCharge*thePtCode==o.theCharge*o.thePtCode && theEtaCode<o.theEtaCode) return true;
    else return false;
  }
   
  bool operator==(const Key& o) const {
    return theEtaCode==o.theEtaCode && thePtCode==o.thePtCode && theCharge==o.theCharge;
  }
  
  friend std::ostream & operator << (std::ostream &out, const Key & o) {
    out << "Key: (eta="<<o.theEtaCode<<", pt="<<o.thePtCode<<", charge="<<o.theCharge<<")";
    return out;
  }

  int theEtaCode;
  unsigned int thePtCode; 
  int          theCharge;

 };
//////////////////////////////////
// Golden Pattern
//////////////////////////////////
class OMTFinput;

class GoldenPattern {

 public:

  typedef std::vector<int> vector1D;
  typedef std::vector<vector1D> vector2D;
  typedef std::vector<vector2D> vector3D;
  typedef std::pair<int,bool> layerResult;

  //
  // GoldenPatterns methods
  //
 GoldenPattern(const Key & aKey, const OMTFConfiguration * omtfConfig) : theKey(aKey), myOmtfConfig(omtfConfig){}
  
  Key key() const {return theKey;}

  void setMeanDistPhi(const vector2D & aMeanDistPhi){ meanDistPhi = aMeanDistPhi; }

  const vector2D & getMeanDistPhi() const {return meanDistPhi;}

  const vector3D & getPdf() const {return pdfAllRef;}

  void setPdf(const vector3D & aPdf){  pdfAllRef = aPdf; }

  int meanDistPhiValue(unsigned int iLayer, unsigned int iRefLayer) const { return meanDistPhi[iLayer][iRefLayer];}

  int pdfValue(unsigned int iLayer, unsigned int iRefLayer, unsigned int iBin) const {return pdfAllRef[iLayer][iRefLayer][iBin];}

  ///Process single measurement layer with a single ref layer
  ///Method should be thread safe
  GoldenPattern::layerResult process1Layer1RefLayer(unsigned int iRefLayer,
						    unsigned int iLayer,
						    const int refPhi,
						    const OMTFinput::vector1D & layerHits);

  friend std::ostream & operator << (std::ostream &out, const GoldenPattern & aPattern);
    
  ///Add a single count to the relevant pdf bin in three dimensions
  void addCount(unsigned int iRefLayer,
		unsigned int iLayer,
		const int refPhi,
		const OMTFinput::vector1D & layerHits);

  ///Reset contents of all data vectors, keeping the vectors size
  void reset();

  ///Normalise event counts in mean dist phi, and pdf vectors to get
  ///the real values of meand dist phi and probability
  void normalise();

  ///Propagate phi from given reference layer to MB2 or ME2
  ///ME2 is used if eta of reference hit is larger than 1.1
  ///expressed in ingerer MicroGMT scale: 1.1/2.61*240 = 101
  int propagateRefPhi(int phiRef, int etaRef, unsigned int iRefLayer);

  ///Check if the GP has any counts in any of referecne layers;
  bool hasCounts();

 private:

  ///Pattern kinematical identification (iEta,iPt,iCharge)
  Key theKey;
  
  ///Distributions for all reference layers
  ///First index: measurement layer number
  ///Second index: refLayer number
  ///Third index: pdf bin number within layer 
  vector3D pdfAllRef;

  ///Mean positions in each layer
  ///First index: measurement layer number 
  ///Second index: refLayer number
  vector2D meanDistPhi;

  ///Vector holding number of counts.
  ///Used for making the patterns
  vector2D meanDistPhiCounts;

  const OMTFConfiguration  * myOmtfConfig;

};
//////////////////////////////////
//////////////////////////////////
#endif 
