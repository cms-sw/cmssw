#ifndef DataFormats_FEDRawData_fed_header_h
#define DataFormats_FEDRawData_fed_header_h

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/*************************************************************************
 *
 * data structures and associated typedefs
 *
 *************************************************************************/

/*
 * FED header - in front of each FED block
 */

typedef struct fedh_struct {
  uint32_t sourceid;
  uint32_t eventid;
} fedh_t;

#define FED_SLINK_START_MARKER 0x5

#define FED_HCTRLID_WIDTH 0x0000000f
#define FED_HCTRLID_SHIFT 28
#define FED_HCTRLID_MASK (FED_HCTRLID_WIDTH << FED_HCTRLID_SHIFT)
#define FED_HCTRLID_EXTRACT(a) (((a) >> FED_HCTRLID_SHIFT) & FED_HCTRLID_WIDTH)

#define FED_EVTY_WIDTH 0x0000000f
#define FED_EVTY_SHIFT 24
#define FED_EVTY_MASK (FED_EVTY_WIDTH << FED_EVTY_SHIFT)
#define FED_EVTY_EXTRACT(a) (((a) >> FED_EVTY_SHIFT) & FED_EVTY_WIDTH)

#define FED_LVL1_WIDTH 0x00ffffff
#define FED_LVL1_SHIFT 0
#define FED_LVL1_MASK (FED_LVL1_WIDTH << FED_LVL1_SHIFT)
#define FED_LVL1_EXTRACT(a) (((a) >> FED_LVL1_SHIFT) & FED_LVL1_WIDTH)

#define FED_BXID_WIDTH 0x00000fff
#define FED_BXID_SHIFT 20
#define FED_BXID_MASK (FED_BXID_WIDTH << FED_BXID_SHIFT)
#define FED_BXID_EXTRACT(a) (((a) >> FED_BXID_SHIFT) & FED_BXID_WIDTH)

#define FED_SOID_WIDTH 0x00000fff
#define FED_SOID_SHIFT 8
#define FED_SOID_MASK (FED_SOID_WIDTH << FED_SOID_SHIFT)
#define FED_SOID_EXTRACT(a) (((a) >> FED_SOID_SHIFT) & FED_SOID_WIDTH)

#define FED_VERSION_WIDTH 0x0000000f
#define FED_VERSION_SHIFT 4
#define FED_VERSION_MASK (FED_VERSION_WIDTH << FED_VERSION_SHIFT)
#define FED_VERSION_EXTRACT(a) (((a) >> FED_VERSION_SHIFT) & FED_VERSION_WIDTH)

#define FED_MORE_HEADERS_WIDTH 0x00000001
#define FED_MORE_HEADERS_SHIFT 3
#define FED_MORE_HEADERS_MASK (FED_MORE_HEADERS_WIDTH << FED_MORE_HEADERS_SHIFT)
#define FED_MORE_HEADERS_EXTRACT(a) (((a) >> FED_MORE_HEADERS_SHIFT) & FED_MORE_HEADERS_WIDTH)

#ifdef __cplusplus
}
#endif

#endif  // DataFormats_FEDRawData_fed_header_h
