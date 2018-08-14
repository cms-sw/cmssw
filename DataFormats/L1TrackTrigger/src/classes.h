/*! \brief   Definition of all the relevant data types
 *  \details Herw we declare instances of all the relevant types. 
 *
 *  \author Andrew W. Rose
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 12
 *
 */

#ifndef L1TRACKTRIGGER_CLASSES_H
#define L1TRACKTRIGGER_CLASSES_H

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace DataFormats_L1TrackTrigger
{
  struct dictionary1 {
    /// Main template type
    Ref_Phase2TrackerDigi_  PD;
    std::vector< Ref_Phase2TrackerDigi_ >                              V_PD;
  };

  struct dictionary2 {
    /// TTCluster and containers
    TTCluster< Ref_Phase2TrackerDigi_ >                                               C_PD;
    std::vector< TTCluster< Ref_Phase2TrackerDigi_ > >                              V_C_PD;
    edm::Wrapper< std::vector< TTCluster< Ref_Phase2TrackerDigi_ > > >            W_V_C_PD;
    edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > >                   SDV_C_PD;
    edm::Wrapper< edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > > > W_SDV_C_PD;

    /// edm::Ref to TTCluster in edmNew::DetSetVector and containers
    edm::Ref< edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > >, TTCluster< Ref_Phase2TrackerDigi_ > >                                    R_C_PD;
    edm::Wrapper< edm::Ref< edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > >, TTCluster< Ref_Phase2TrackerDigi_ > > >                  W_R_C_PD;
    std::vector< edm::Ref< edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > >, TTCluster< Ref_Phase2TrackerDigi_ > > >                   V_R_C_PD;
    edm::Wrapper< std::vector< edm::Ref< edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > >, TTCluster< Ref_Phase2TrackerDigi_ > > > > W_V_R_C_PD;
  };

  struct dictionary3 {
    /// TTStub and containers
    TTStub< Ref_Phase2TrackerDigi_ >                                               S_PD;
    std::vector< TTStub< Ref_Phase2TrackerDigi_ > >                              V_S_PD;
    edm::Wrapper< std::vector< TTStub< Ref_Phase2TrackerDigi_ > > >            W_V_S_PD;
    edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >                   SDV_S_PD;
    edm::Wrapper< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > > > W_SDV_S_PD;

    /// edm::Ref to TTStub in edmNew::DetSetVector and containers
    edm::Ref< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >, TTStub< Ref_Phase2TrackerDigi_ > >                                    R_S_PD;
    edm::Wrapper< edm::Ref< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >, TTStub< Ref_Phase2TrackerDigi_ > > >                  W_R_S_PD;
    std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >, TTStub< Ref_Phase2TrackerDigi_ > > >                   V_R_S_PD;
    edm::Wrapper< std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_Phase2TrackerDigi_ > >, TTStub< Ref_Phase2TrackerDigi_ > > > > W_V_R_S_PD;
  };


  struct dictionarytrack {
    /// TTTrack and containers
    TTTrack< Ref_Phase2TrackerDigi_ >                                               T_PD;
    std::vector< TTTrack< Ref_Phase2TrackerDigi_ > >                              V_T_PD;
    edm::Wrapper< std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > >            W_V_T_PD;
    edmNew::DetSetVector< TTTrack< Ref_Phase2TrackerDigi_ > >                   SDV_T_PD;
    edm::Wrapper< edmNew::DetSetVector< TTTrack< Ref_Phase2TrackerDigi_ > > > W_SDV_T_PD;

    /// edm::Ref to TTTrack in edmNew::DetSetVector and containers
    edm::Ref< edmNew::DetSetVector< TTTrack< Ref_Phase2TrackerDigi_ > >, TTTrack< Ref_Phase2TrackerDigi_ > >                                    R_T_PD;
    edm::Wrapper< edm::Ref< edmNew::DetSetVector< TTTrack< Ref_Phase2TrackerDigi_ > >, TTTrack< Ref_Phase2TrackerDigi_ > > >                  W_R_T_PD;
    std::vector< edm::Ref< edmNew::DetSetVector< TTTrack< Ref_Phase2TrackerDigi_ > >, TTTrack< Ref_Phase2TrackerDigi_ > > >                   V_R_T_PD;
    edm::Wrapper< std::vector< edm::Ref< edmNew::DetSetVector< TTTrack< Ref_Phase2TrackerDigi_ > >, TTTrack< Ref_Phase2TrackerDigi_ > > > > W_V_R_T_PD;
  };
}

#endif
