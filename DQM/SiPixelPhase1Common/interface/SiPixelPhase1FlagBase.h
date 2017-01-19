#ifndef SiStrip_Phase1FlagBase_h
#define SiStrip_Phase1FlagBase_h

/*******************************************************************************
*
*  Filename    : SiPixelPhaseIFlagBase.h
*  Description :
*     An extended base class from the SiPixelPhaseIBase class to allow for
*     TriggerEventFlag filtering.
*
*  Author      : Yi-Mu "Enoch" Chen [ ensc@hep1.phys.ntu.edu.tw ]
*
*******************************************************************************/
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"

#include <vector>
#include <memory>

class SiPixelPhase1FlagBase : public SiPixelPhase1Base
{
public:
   SiPixelPhase1FlagBase( const edm::ParameterSet& iConfig );
   virtual ~SiPixelPhase1FlagBase ();

   // GenericTriggerEventFlag requires additional setup
   void dqmBeginRun( const edm::Run&, const edm::EventSetup& );

   // Overloading the analyze function to include filter checking
   void analyze( const edm::Event&, const edm::EventSetup& );

   // Pure virtual function for users to over load
   virtual void flagAnalyze( const edm::Event&, const edm::EventSetup& ) = 0;

private:
   typedef std::unique_ptr<GenericTriggerEventFlag> FlagPtr;
   std::vector<FlagPtr> _flaglist;
};


#endif/* end of include guard: SiStripPhaseIFlagBase */
