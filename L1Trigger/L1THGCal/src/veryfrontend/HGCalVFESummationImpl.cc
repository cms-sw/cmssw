#include "L1Trigger/L1THGCal/interface/veryfrontend/HGCalVFESummationImpl.h"

HGCalVFESummationImpl::
HGCalVFESummationImpl(const edm::ParameterSet& conf):
  thickness_corrections_(conf.getParameter<std::vector<double>>("ThicknessCorrections"))
{}

void 
HGCalVFESummationImpl::
triggerCellSums(const HGCalTriggerGeometryBase& geometry, 
                const std::vector<std::pair<DetId, uint32_t > >& linearized_dataframes,
                std::map<HGCalDetId, uint32_t>& payload)
{
  if(linearized_dataframes.empty()) return;
  // sum energies in trigger cells
  for(const auto& frame : linearized_dataframes)
  {
    DetId cellid(frame.first);

    // find trigger cell associated to cell
    uint32_t tcid = geometry.getTriggerCellFromCell(cellid);
    HGCalDetId triggercellid( tcid );
    payload.insert( std::make_pair(triggercellid, 0) ); // do nothing if key exists already
    uint32_t value = frame.second;
    // equalize value among cell thicknesses
    if(cellid.det()==DetId::Forward)
    {
      int thickness = 0;
      switch(cellid.subdetId())
      { 
        case ForwardSubdetector::HGCEE:
          thickness = geometry.eeTopology().dddConstants().waferTypeL(HGCalDetId(cellid).wafer())-1;
          break;
        case ForwardSubdetector::HGCHEF:
          thickness = geometry.fhTopology().dddConstants().waferTypeL(HGCalDetId(cellid).wafer())-1;
          break;
        default:
          break;
      };

      double thickness_correction = thickness_corrections_.at(thickness);
      value = (double)value*thickness_correction;
    }

    // sums energy for the same triggercellid
    payload[triggercellid] += value; // 32 bits integer should be largely enough 
  }

}
