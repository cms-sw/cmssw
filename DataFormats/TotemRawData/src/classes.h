/****************************************************************************
*
* This is a part of the TOTEM testbeam/monitoring software.
* This is a part of the TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*    
****************************************************************************/

#include "DataFormats/TotemRawData/interface/TotemRawEvent.h"
#include "DataFormats/TotemRawData/interface/TotemRawToDigiStatus.h"

#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    TotemRawEvent dummy10;
    edm::Wrapper<TotemRawEvent> dummy11;

    TotemRawEvent::OptoRxMetaData dummy20;

    TotemRawEvent::TriggerData dummy25;

    TotemStructuralVFATId dummy28;

    TotemRawToDigiStatus dummy30;
    edm::Wrapper< TotemRawToDigiStatus > dummy31;

    TotemVFATStatus dummy40;
    edm::Wrapper< TotemVFATStatus > dummy41;

    std::bitset<8> dummy50;
    edm::Wrapper< std::bitset<8> > dummy51;
  }
}
