/*******************************************************************************
*
*  Package : SiPixelPhase1Common
*  Class   : SiPixelPhase1FlagBase
*
*  Implementations for trigger event flag filtering
*
*  Author      : Yi-Mu "Enoch" Chen [ ensc@hep1.phys.ntu.edu.tw ]
*
*******************************************************************************/

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1FlagBase.h"

/*******************************************************************************
*   Constructor and Destructor
*******************************************************************************/
SiPixelPhase1FlagBase::SiPixelPhase1FlagBase( const edm::ParameterSet& iConfig ) :
   SiPixelPhase1Base( iConfig )
{
   for( const auto& pset : iConfig.getParameter<std::vector<edm::ParameterSet> >( "flaglist" ) ){
      _flaglist.emplace_back( new GenericTriggerEventFlag(pset, consumesCollector(), *this)  );
   }
}

SiPixelPhase1FlagBase::~SiPixelPhase1FlagBase(){}

/*******************************************************************************
*   GenericTriggerEventFlag initialization
*******************************************************************************/
void
SiPixelPhase1FlagBase::dqmBeginRun( const edm::Run& iRun, const edm::EventSetup& iSetup )
{
   for( auto& flag : _flaglist ){
      if( flag->on() ){ flag->initRun( iRun, iSetup ); }
   }
}

/*******************************************************************************
*   Overloaded analyze function
*******************************************************************************/
void
SiPixelPhase1FlagBase::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   if( iEvent.isRealData() ){
      // requiring additional filter for real data only
      for( auto& flag : _flaglist ){
         // Early exit if any flag is on but not accepted
         if( flag->on() && !flag->accept( iEvent, iSetup ) ){
            return;
         }
      }
   } else {
      // Enabling flagging for MC sample to debug.
      for( auto& flag : _flaglist ){
         // Early exit if any flag is on but not accepted
         if( flag->on() && !flag->accept( iEvent, iSetup ) ){
            return;
         }
      }
   }

   // Running user defined virtual function
   flagAnalyze( iEvent, iSetup );
}
