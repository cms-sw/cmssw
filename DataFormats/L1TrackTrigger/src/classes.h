/*! \brief   Definition of all the relevant data types
 *  \details Herw we declare instances of all the relevant types. 
 *
 *  \author Andrew W. Rose
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 12
 *
 */

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

namespace
{
  namespace
  {
    /// Main template type
    Ref_PixelDigi_  PD;

    /// TTCluster and containers
    TTCluster< Ref_PixelDigi_ >                                               C_PD;
    std::vector< TTCluster< Ref_PixelDigi_ > >                              V_C_PD;
    edm::Wrapper< std::vector< TTCluster< Ref_PixelDigi_ > > >            W_V_C_PD;
    edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >                   SDV_C_PD;
    edm::Wrapper< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > > > W_SDV_C_PD;

    /// edm::Ref to TTCluster in edmNew::DetSetVector and containers
    edm::Ref< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >, TTCluster< Ref_PixelDigi_ > >                                    R_C_PD;
    edm::Wrapper< edm::Ref< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >, TTCluster< Ref_PixelDigi_ > > >                  W_R_C_PD;
    std::vector< edm::Ref< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >, TTCluster< Ref_PixelDigi_ > > >                   V_R_C_PD;
    edm::Wrapper< std::vector< edm::Ref< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >, TTCluster< Ref_PixelDigi_ > > > > W_V_R_C_PD;

    /// TTStub and containers
    TTStub< Ref_PixelDigi_ >                                               S_PD;
    std::vector< TTStub< Ref_PixelDigi_ > >                              V_S_PD;
    edm::Wrapper< std::vector< TTStub< Ref_PixelDigi_ > > >            W_V_S_PD;
    edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >                   SDV_S_PD;
    edm::Wrapper< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > > > W_SDV_S_PD;

    /// edm::Ref to TTStub in edmNew::DetSetVector and containers
    edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > >                                    R_S_PD;
    edm::Wrapper< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > >                  W_R_S_PD;
    std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > >                   V_R_S_PD;
    edm::Wrapper< std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > > W_V_R_S_PD;

    /// TTTrack and containers
    TTTrack< Ref_PixelDigi_ >                                    T_PD;
    std::vector< TTTrack< Ref_PixelDigi_ > >                   V_T_PD;
    edm::Wrapper< std::vector< TTTrack< Ref_PixelDigi_ > > > W_V_T_PD;

    /// edm::Ptr to TTTrack and containers
    edm::Ptr< TTTrack< Ref_PixelDigi_ > >                                    P_T_PD;
    edm::Wrapper< edm::Ptr< TTTrack< Ref_PixelDigi_ > > >                  W_P_T_PD;
    std::vector< edm::Ptr< TTTrack< Ref_PixelDigi_ > > >                   V_P_T_PD;
    edm::Wrapper< std::vector< edm::Ptr< TTTrack< Ref_PixelDigi_ > > > > W_V_P_T_PD;

  }
}

