#ifndef DataFormats_FEDRawData_fed_trailer_h
#define DataFormats_FEDRawData_fed_trailer_h

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
 * FED trailer - at the end of each FED block
 */

typedef struct fedt_struct {
  uint32_t conscheck;
  uint32_t eventsize;
} fedt_t ;

#define FED_SLINK_END_MARKER         0xa

#define FED_TCTRLID_WIDTH            0x0000000f
#define FED_TCTRLID_SHIFT            28
#define FED_TCTRLID_MASK             ( FED_TCTRLID_WIDTH << FED_TCTRLID_SHIFT )
#define FED_TCTRLID_EXTRACT(a)       ( ( (a) >> FED_TCTRLID_SHIFT ) & FED_TCTRLID_WIDTH )

#define FED_EVSZ_WIDTH               0x00ffffff
#define FED_EVSZ_SHIFT               0
#define FED_EVSZ_MASK                ( FED_EVSZ_WIDTH << FED_EVSZ_SHIFT )
#define FED_EVSZ_EXTRACT(a)          ( ( (a) >> FED_EVSZ_SHIFT ) & FED_EVSZ_WIDTH )

#define FED_CRCS_WIDTH               0x0000ffff
#define FED_CRCS_SHIFT               16
#define FED_CRCS_MASK                ( FED_CRCS_WIDTH << FED_CRCS_SHIFT )
#define FED_CRCS_EXTRACT(a)          ( ( (a) >> FED_CRCS_SHIFT ) & FED_CRCS_WIDTH )

#define FED_STAT_WIDTH               0x0000000f
#define FED_STAT_SHIFT               8
#define FED_STAT_MASK                ( FED_STAT_WIDTH << FED_STAT_SHIFT )
#define FED_STAT_EXTRACT(a)          ( ( (a) >> FED_STAT_SHIFT ) & FED_STAT_WIDTH )

#define FED_TTSI_WIDTH               0x0000000f
#define FED_TTSI_SHIFT               4
#define FED_TTSI_MASK                ( FED_TTSI_WIDTH << FED_TTSI_SHIFT )
#define FED_TTSI_EXTRACT(a)          ( ( (a) >> FED_TTSI_SHIFT ) & FED_TTSI_WIDTH )

#define FED_MORE_TRAILERS_WIDTH      0x00000001
#define FED_MORE_TRAILERS_SHIFT      3
#define FED_MORE_TRAILERS_MASK       ( FED_MORE_TRAILERS_WIDTH << FED_MORE_TRAILERS_SHIFT )
#define FED_MORE_TRAILERS_EXTRACT(a) ( ( (a) >> FED_MORE_TRAILERS_SHIFT ) & FED_MORE_TRAILERS_WIDTH )

#define FED_CRC_MODIFIED_WIDTH       0x00000001
#define FED_CRC_MODIFIED_SHIFT       2
#define FED_CRC_MODIFIED_MASK        ( FED_CRC_MODIFIED_WIDTH << FED_CRC_MODIFIED_SHIFT )
#define FED_CRC_MODIFIED_EXTRACT(a)  ( ( (a) >> FED_CRC_MODIFIED_SHIFT ) & FED_CRC_MODIFIED_WIDTH )

#define FED_SLINK_ERROR_WIDTH        0x00000001
#define FED_SLINK_ERROR_SHIFT        14
#define FED_SLINK_ERROR_MASK         ( FED_SLINK_ERROR_WIDTH << FED_SLINK_ERROR_SHIFT )
#define FED_SLINK_ERROR_EXTRACT(a)   ( ( (a) >> FED_SLINK_ERROR_SHIFT ) & FED_SLINK_ERROR_WIDTH )

#define FED_WRONG_FEDID_WIDTH        0x00000001
#define FED_WRONG_FEDID_SHIFT        15
#define FED_WRONG_FEDID_MASK         ( FED_WRONG_FEDID_WIDTH << FED_WRONG_FEDID_SHIFT )
#define FED_WRONG_FEDID_EXTRACT(a)   ( ( (a) >> FED_WRONG_FEDID_SHIFT ) & FED_WRONG_FEDID_WIDTH )

#ifdef __cplusplus
}
#endif

#endif // DataFormats_FEDRawData_fed_trailer_h
