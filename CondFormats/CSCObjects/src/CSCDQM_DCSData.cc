/*
 * =====================================================================================
 *
 *       Filename:  CSCDQM_DCSData.cc
 *
 *    Description:  CSCDQM DCS Data Object implementation
 *
 *        Version:  1.0
 *        Created:  05/04/2009 11:38:01 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), valdas.rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "CondFormats/CSCObjects/interface/CSCDQM_DCSData.h"

namespace cscdqm {

  DCSData::DCSData() {
    iov = 0;
    last_change = 0;
    temp_mode = 0;
    hvv_mode = 0;
    lvv_mode = true;
    lvi_mode = 0.0;
  }

  DCSData::~DCSData() {}

}  // namespace cscdqm
