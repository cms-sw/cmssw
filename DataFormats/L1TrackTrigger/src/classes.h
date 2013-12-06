/*! \brief   Definition of all the relevant data types
 *  \details Herw we declare instances of all the relevant types. 
 *
 *  \author Andrew W. Rose
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 12
 *
 */

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

namespace
{
  namespace
  {
    Ref_PixelDigi_  PD;

    TTCluster< Ref_PixelDigi_ >                                                  C_PD;
    std::vector< TTCluster< Ref_PixelDigi_ > >                                 V_C_PD;
    edm::Wrapper< std::vector< TTCluster< Ref_PixelDigi_ > > >               W_V_C_PD;
    edm::Ptr< TTCluster< Ref_PixelDigi_ > >                                    P_C_PD;
    edm::Wrapper< edm::Ptr< TTCluster< Ref_PixelDigi_ > > >                  W_P_C_PD;
    std::vector< edm::Ptr< TTCluster< Ref_PixelDigi_ > > >                   V_P_C_PD;
    edm::Wrapper< std::vector< edm::Ptr< TTCluster< Ref_PixelDigi_ > > > > W_V_P_C_PD;

    TTStub< Ref_PixelDigi_ >                                                  S_PD;
    std::vector< TTStub< Ref_PixelDigi_ > >                                 V_S_PD;
    edm::Wrapper< std::vector< TTStub< Ref_PixelDigi_ > > >               W_V_S_PD;
    edm::Ptr< TTStub< Ref_PixelDigi_ > >                                    P_S_PD;
    edm::Wrapper< edm::Ptr< TTStub< Ref_PixelDigi_ > > >                  W_P_S_PD;
    std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > >                   V_P_S_PD;
    edm::Wrapper< std::vector< edm::Ptr< TTStub< Ref_PixelDigi_ > > > > W_V_P_S_PD;

    TTTrack< Ref_PixelDigi_ >                                                  T_PD;
    std::vector< TTTrack< Ref_PixelDigi_ > >                                 V_T_PD;
    edm::Wrapper< std::vector< TTTrack< Ref_PixelDigi_ > > >               W_V_T_PD;
    edm::Ptr< TTTrack< Ref_PixelDigi_ > >                                    P_T_PD;
    edm::Wrapper< edm::Ptr< TTTrack< Ref_PixelDigi_ > > >                  W_P_T_PD;
    std::vector< edm::Ptr< TTTrack< Ref_PixelDigi_ > > >                   V_P_T_PD;
    edm::Wrapper< std::vector< edm::Ptr< TTTrack< Ref_PixelDigi_ > > > > W_V_P_T_PD;
  }
}

