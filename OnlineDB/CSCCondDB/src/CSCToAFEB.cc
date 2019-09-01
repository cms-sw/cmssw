/**\class  CSCToAFEB
 *
 * CSC layer, wire vs AFEB channel map. 
 *
 * \author N. Terentiev, CMU
 */

#include <OnlineDB/CSCCondDB/interface/CSCToAFEB.h>
#include <iostream>

using namespace std;

/// return AFEB channel number
int CSCToAFEB::getAfebCh(int layer, int wiregroup) const {
  int wire = wiregroup - 8 * ((wiregroup - 1) / 8);
  int channel = layer_wire_to_channel_[layer - 1][wire - 1];
  return channel;
}
/// return AFEB position number
int CSCToAFEB::getAfebPos(int layer, int wiregroup) const {
  int col = (wiregroup - 1) / 8 + 1;
  int wire = wiregroup - 8 * ((wiregroup - 1) / 8);
  int afeb = (col - 1) * 3 + layer_wire_to_board_[layer - 1][wire - 1];
  return afeb;
}
/// return layer number
int CSCToAFEB::getLayer(int afeb, int channel) const {
  int col = (afeb - 1) / 3 + 1;
  int pos_in_col = afeb - (col - 1) * 3;
  int layer = pos_in_col * 2 - 1;
  if (channel < 5 || (channel > 8 && channel < 13))
    layer++;
  return layer;
}
/// return wiregroup number
int CSCToAFEB::getWireGroup(int afeb, int channel) const {
  int col = (afeb - 1) / 3 + 1;
  int wire = (col - 1) * 8 + 1;
  if (channel < 5)
    wire = wire + (channel - 1);
  if (channel > 4 && channel < 9)
    wire = wire + (channel - 5);
  if (channel > 8 && channel < 13)
    wire = wire + (channel - 5);
  if (channel > 12)
    wire = wire + (channel - 9);
  return wire;
}
/// return max. number of AFEBs
int CSCToAFEB::getMaxAfeb(int station, int ring) const { return station_ring_to_nmxafeb_[station - 1][ring - 1]; }

/// return max. number of wiregroups per plane
int CSCToAFEB::getMaxWire(int station, int ring) const { return station_ring_to_nmxwire_[station - 1][ring - 1]; }
