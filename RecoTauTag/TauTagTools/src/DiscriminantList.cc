#include "RecoTauTag/TauTagTools/interface/DiscriminantList.h"

using namespace std;
namespace PFTauDiscriminants {

// CONSTRUCTOR:
//      USER: add/remove the Discrimiants* you wish to use here.
DiscriminantList::DiscriminantList()
{
   theDiscriminants_.push_back(new DecayMode()                         ); 
   theDiscriminants_.push_back(new MainTrackPt()                       ); 
   theDiscriminants_.push_back(new MainTrackAngle()                    ); 
   theDiscriminants_.push_back(new TrackPt()                           ); 
   theDiscriminants_.push_back(new TrackAngle()                        ); 
   theDiscriminants_.push_back(new PiZeroPt()                          ); 
   theDiscriminants_.push_back(new PiZeroAngle()                       ); 
   theDiscriminants_.push_back(new Dalitz()                            ); 
   theDiscriminants_.push_back(new InvariantMassOfSignal()             ); 
   theDiscriminants_.push_back(new InvariantMass()                     ); 
   theDiscriminants_.push_back(new Pt()                                ); 
   theDiscriminants_.push_back(new Eta()                               ); 
   theDiscriminants_.push_back(new OutlierPt()                         ); 
   theDiscriminants_.push_back(new OutlierAngle()                      ); 
   theDiscriminants_.push_back(new ChargedOutlierPt()                  ); 
   theDiscriminants_.push_back(new ChargedOutlierAngle()               ); 
   theDiscriminants_.push_back(new NeutralOutlierPt()                  ); 
   theDiscriminants_.push_back(new NeutralOutlierAngle()               ); 
   theDiscriminants_.push_back(new OutlierNCharged()                   ); 
   theDiscriminants_.push_back(new GammaOccupancy()                    ); 
   theDiscriminants_.push_back(new GammaPt()                           ); 
   theDiscriminants_.push_back(new FilteredObjectPt()                  ); 
   theDiscriminants_.push_back(new InvariantMassOfSignalWithFiltered() ); 
   theDiscriminants_.push_back(new OutlierN()                          ); 
   theDiscriminants_.push_back(new OutlierSumPt()                      ); 
   theDiscriminants_.push_back(new OutlierMass()                       ); 
   theDiscriminants_.push_back(new ChargedOutlierSumPt()               ); 
   theDiscriminants_.push_back(new NeutralOutlierSumPt()               ); 
}

//cleanup on destruction
DiscriminantList::~DiscriminantList()
{
   for(const_iterator iDiscrminant  = this->begin();
                      iDiscrminant != this->end();
                    ++iDiscrminant)
   {
      delete *iDiscrminant;
   }
}

} //end namespace



     
