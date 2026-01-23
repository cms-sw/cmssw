#include "Geometry/CaloTopology/interface/HGCalNeighbourFinder.h"

constexpr int densityNumberLD = 8;
constexpr int densityNumberHD = 12;

constexpr unsigned int iuMask = 0x0000001F;
constexpr unsigned int ivMask = 0x000003E0;
constexpr unsigned int waferMask = 0x000FFC00;
constexpr unsigned int layerMask = 0x01F00000;
constexpr unsigned int detectorMask = 0xF0000000;

constexpr unsigned int HGCalEE = 0x80000000;
constexpr unsigned int HGCalHSi = 0x90000000;

constexpr unsigned int signMask = 0x00000010;

constexpr int ivShift = 5;
constexpr int waferShift = 10;

constexpr int duCell[6] = {-1, 0, +1, +1, 0, -1};
constexpr int dvCell[6] = {-1, -1, 0, +1, +1, 0};
constexpr int duWaf[6] = {0, +1, +1, 0, -1, -1};
const int dvWaf[6] = {-1, 0, +1, +1, 0, -1};

HGCalNeighbourFinder::HGCalNeighbourFinder(const HGCalDDDConstants* hgc) : hgc_(hgc) {
  /* ----------------------------------------
     Fill the edgeIndex -> iu,iv mappings
     ---------------------------------------- */
  for (int iu = 0; iu < 2 * densityNumberLD; iu++) {
    for (int iv = 0; iv < 2 * densityNumberLD; iv++) {
      int edgeIndex = edgeIndexForU(iu, iv, false);
      if (edgeIndex > -1) {
        iuEdgeLD[edgeIndex] = iu;
        ivEdgeLD[edgeIndex] = iv;
      }
    }
  }

  for (int iu = 0; iu < 2 * densityNumberHD; iu++) {
    for (int iv = 0; iv < 2 * densityNumberHD; iv++) {
      int edgeIndex = edgeIndexForU(iu, iv, true);
      if (edgeIndex > -1) {
        iuEdgeHD[edgeIndex] = iu;
        ivEdgeHD[edgeIndex] = iv;
      }
    }
  }

  /* ----------------------------------------------
     Fill the edgeIndex -> side mappings
     and the edgeIndex -> corner mappings
     ---------------------------------------------- */
  int edgeIndex = 1;
  int edgeCount = densityNumberLD - 1;
  int nedge = 6 * densityNumberLD - 3;
  for (int i = 0; i < 6; i++) {
    for (int j = edgeIndex; j < edgeIndex + edgeCount + i % 2; j++) {
      sideLD[j % nedge] = i;
    }
    edgeIndex += edgeCount + i % 2;
  }

  edgeIndex = 0;
  for (int i = 0; i < 6; i++) {
    sideLD[edgeIndex] += (i + 1) * 10;
    edgeIndex += edgeCount + (i + 1) % 2;
  }

  edgeIndex = 1;
  edgeCount = densityNumberHD - 1;
  nedge = 6 * densityNumberHD - 3;
  for (int i = 0; i < 6; i++) {
    for (int j = edgeIndex; j < edgeIndex + edgeCount + i % 2; j++) {
      sideHD[j % nedge] = i;
    }
    edgeIndex += edgeCount + i % 2;
  }

  edgeIndex = 0;
  for (int i = 0; i < 6; i++) {
    sideHD[edgeIndex] += (i + 1) * 10;
    edgeIndex += edgeCount + (i + 1) % 2;
  }
}

int HGCalNeighbourFinder::edgeIndexForU(int iu, int iv, bool hd) const {
  int densityNumber = (hd) ? densityNumberHD : densityNumberLD;

  int maxIndex = 2 * densityNumber - 1;
  int halfMax = densityNumber - 1;

  if ((iv > (iu + halfMax)) || (iv < (iu - densityNumber)))
    return -1;  // iu:iv for non-existent cell

  int edgeIndex = -1;

  if ((iv == 0) || (iu - iv == densityNumber))
    edgeIndex = iu;
  else if (iu == maxIndex)
    edgeIndex = maxIndex + iv - halfMax;
  else if (iv == maxIndex)
    edgeIndex = 2 * maxIndex + densityNumber - iu;
  else if (iv - iu == halfMax)
    edgeIndex = 2 * maxIndex + densityNumber - iu;
  else if (iu == 0)
    edgeIndex = 3 * maxIndex - iv;

  return edgeIndex;
}

std::vector<unsigned int> HGCalNeighbourFinder::nearestNeighboursOfDetId(unsigned int detId) const {
  std::vector<unsigned int> detIdVec(8, 0);
  if (!((detId & detectorMask) == HGCalEE || (detId & detectorMask) == HGCalHSi))
    return detIdVec;
  HGCSiliconDetId id(detId);

  int layer = id.layer();
  int iu = id.waferU();
  int iv = id.waferV();
  bool HD = hgc_->waferIsHD(layer, iu, iv);
  int edgeIndex = edgeIndexForU(iu, iv, HD);
  bool partialWafer = hgc_->waferPartial(layer, iu, iv);

  if (edgeIndex < 0) {  // Cell is not on the edge of a wafer (~80% of cells)
    if (partialWafer) {
      // Special treatment for partial wafers: some cells present in whole wafers do not exist
      int nn = 0;
      for (int i = 0; i < 6; i++) {
        detIdVec[nn] = (detId & ~(iuMask | ivMask)) | (iu + duCell[i]) | ((iv + dvCell[i]) << ivShift);
        if (hgc_->waferExist(layer, (iu + duCell[i]), (iv + dvCell[i])))
          nn++;
        else
          detIdVec[nn] = 0;
      }
    } else {
      detIdVec[0] = (detId & ~(iuMask | ivMask)) | (iu - 1) | ((iv - 1) << ivShift);
      detIdVec[1] = (detId & ~(iuMask | ivMask)) | (iu) | ((iv - 1) << ivShift);
      detIdVec[2] = (detId & ~(iuMask | ivMask)) | (iu + 1) | ((iv) << ivShift);
      detIdVec[3] = (detId & ~(iuMask | ivMask)) | (iu + 1) | ((iv + 1) << ivShift);
      detIdVec[4] = (detId & ~(iuMask | ivMask)) | (iu) | ((iv + 1) << ivShift);
      detIdVec[5] = (detId & ~(iuMask | ivMask)) | (iu - 1) | ((iv) << ivShift);
    }
  } else {  // Cell is on the edge
    int* iuEdge = (int*)(iuEdgeLD);
    int* ivEdge = (int*)(ivEdgeLD);
    int* side = (int*)(sideLD);
    int densityNumber = densityNumberLD;
    if (HD) {
      iuEdge = (int*)(iuEdgeHD);
      ivEdge = (int*)(ivEdgeHD);
      side = (int*)(sideHD);
      densityNumber = densityNumberHD;
    }

    int edgeCount = 3 * (2 * densityNumber - 1);
    int mod = 2 * densityNumber;
    int iside = side[edgeIndex] % 10;
    int corner = side[edgeIndex] / 10 - 1;

    /* -------------------------------------------------------------------------------
       First step: include the 4 neighbours in the same wafer (corners only 3)
       ------------------------------------------------------------------------------- */
    int icount = 4;
    int ioff = iside + 2;
    if (corner > -1) {
      icount = 3;
      ioff = corner + 2;
    }

    int nn = 0;
    for (int i = 0; i < icount; i++) {
      int j = (ioff + i) % 6;
      detIdVec[nn] =
          (detId & ~(iuMask | ivMask)) | (iu + duCell[j] + mod) % mod | ((iv + dvCell[j] + mod) % mod << ivShift);
      if (partialWafer) {
        if (!(hgc_->waferExist(layer, ((iu + duCell[i] + mod) % mod), ((iv + dvCell[i] + mod) % mod)))) {
          detIdVec[nn] = 0;
          nn--;
        }
      }
      nn++;
    }
    icount = nn;
    /* --------------------------------------------------------------------------------
       There is a special case in partial wafer LD 1 (Top Half) where edgeIndex = 37 is
       not an edge cell. The result is slightly ugly but not crazy.
       It could be corrected for here by code like:
       const int weirdPartialCell = 37;
       if(partialWafer) {
       if([theDetInterface partialType] == 1 && edgeIndex == weirdPartialCell) return;
       }
       -------------------------------------------------------------------------------- */

    /* -------------------------------------------------------------------------------
       Second step: Find the wafer adjacent to this wafer side
       ------------------------------------------------------------------------------- */
    bool mirror = false;
    int irot = hgc_->placementIndex(id);  //[theDetInterface placementIndexForWafer:DetId];
    int idir = (iside + irot) % 6;
    if (irot > 5) {
      mirror = true;
      irot = (12 - irot) % 6;
      idir = (irot - iside + 5) % 6;
    }

    int waferId = (detId & waferMask) >> waferShift;

    int wiu = waferId & iuMask;
    int wiv = (waferId & ivMask) >> ivShift;

    if (wiu & signMask)
      wiu = -(wiu & ~signMask);

    if (wiv & signMask)
      wiv = -(wiv & ~signMask);

    int wiuNxt = wiu + duWaf[idir];
    int wivNxt = wiv + dvWaf[idir];

    int wuId = abs(wiuNxt);
    if (wiuNxt < 0)
      wuId = wuId | signMask;
    int wvId = abs(wivNxt);
    if (wivNxt < 0)
      wvId = wvId | signMask;

    int detIdNxt = (detId & ~(waferMask)) | (wuId | (wvId << ivShift)) << waferShift;
    HGCSiliconDetId idNxt(detIdNxt);

    // Next wafer adjacent to this edge may not exist
    // (We could be on the edge of the HGCAL acceptance)
    // if so, we are done...
    if (!(hgc_->waferExist(idNxt.layer(), idNxt.waferU(), idNxt.waferV())))
      return detIdVec;

    /* -------------------------------------------------------------------------------
       Third step: locate the neighbour cells in the wafer specified by DetIdNxt
       ------------------------------------------------------------------------------- */
    int jrot = hgc_->placementIndex(idNxt);
    if (jrot > 5)
      jrot = (12 - jrot) % 6;

    int drot = (irot - jrot + 6) % 6;
    if (mirror)
      drot = (6 - drot) % 6;

    bool HDnxt = hgc_->waferIsHD(idNxt.layer(), idNxt.waferU(), idNxt.waferV());
    bool sameDens = (HD == HDnxt);

    int maxIndex = 2 * densityNumber - 1;
    int sum, newIndex, istart, iend;

    if (drot % 2 == 0) {
      // --- Ideally matched neighbour wafer (extended edge cells with truncated edge cells)
      sum = maxIndex * ((iside + 2) % 3);
      newIndex = (sum - edgeIndex + (drot / 2) * maxIndex + edgeCount) % edgeCount;
      istart = 0;
      iend = 2;
    } else {
      // --- Imperfectly matched neighbour wafer (extended edge cells with extended edge cells,
      //      or truncated with trucated)
      sum = ((densityNumber - 1) + (iside + 4) * maxIndex) % (3 * maxIndex);
      newIndex = (sum - edgeIndex + (drot / 2 + 1) * maxIndex + edgeCount) % edgeCount;
      istart = 0;
      iend = 3;
      if (corner > -1) {
        if (corner % 2 == 0) {
          istart = 1;
        } else {
          iend = 2;
        }
      }
    }
    /* ----------------------------------------------------------------------------------------
       Deal now with the special case of crossing to a wafer with different
       density.
       Need to deal with a number of specific cases identified empirically
       and not analytically explicable
       ---------------------------------------------------------------------------------------- */
    if (!sameDens) {  // Wafer density changing
      if (HDnxt) {    // LD -> HD transition
        newIndex = (3 * newIndex) / 2 + drot % 2;
        iuEdge = (int*)(iuEdgeHD);
        ivEdge = (int*)(ivEdgeHD);
        edgeCount = 3 * (2 * densityNumberHD - 1);
        iend = 3;
        int jside = sideHD[newIndex];
        if (corner > -1) {  // ---- Special treatment for LD corners
          if (corner == 0 && jside == 1)
            newIndex++;
          else if (corner == 1 && jside == 1)
            iend = 2;
          else if (corner == 3 && (jside == 3 || jside == 4))
            istart = 1;
          else if (corner == 4 && jside == 3)
            newIndex++;
          else if (corner == 5 && jside == 3)
            istart = 1;
        }
      } else {  // HD -> LD transition
        newIndex = (2 * newIndex) / 3;
        iuEdge = (int*)(iuEdgeLD);
        ivEdge = (int*)(ivEdgeLD);
        edgeCount = 3 * (2 * densityNumberLD - 1);
        iend = 2;
        int jside = sideLD[newIndex] % 10;
        //int jcorner = sideLD[newIndex] / 10 - 1;
        if (iside == 1) {  // Special treatment for HD -> LD transition
          if (jside == 1) {
            if (edgeIndex % 3 == 0)
              newIndex++;
          } else if (jside == 4) {
            if (edgeIndex % 3 == 0)
              newIndex--;
            else if (edgeIndex % 3 == 1)
              iend = 1;
          }
        } else if (iside == 2) {
          if (jside == 2) {
            if (edgeIndex % 3 == 2)
              istart = 1;
          } else if (jside == 4) {
            if (edgeIndex % 3 == 2)
              iend = 1;
          } else if (jside == 5) {
            if (edgeIndex % 3 != 1)
              newIndex--;
          }
        } else if (iside == 3) {
          if (jside == 5) {
            if (edgeIndex % 3 == 1)
              newIndex--;
          }
        } else if (iside == 4) {
          if (jside == 3) {
            if (edgeIndex % 3 == 2)
              newIndex--;
          }
        }
        if (corner > -1) {  // Special treatment for HD corners
          if (corner == 1) {
            if (jside == 5)
              newIndex--;
          } else if (corner == 3)
            iend = 1;
          else if (corner == 4 && jside == 5) {
            istart = 0;
            iend = 1;
          }
        }
      }
    }

    partialWafer = hgc_->waferPartial(idNxt.layer(), idNxt.waferU(), idNxt.waferV());
    // ---- Loop now adds the 1,2 or 3 cells in the adjacent wafer
    for (int i = istart; i < iend; i++) {
      int iuNxt = iuEdge[(newIndex + i) % edgeCount];
      int ivNxt = ivEdge[(newIndex + i) % edgeCount];
      detIdVec[icount] = (detIdNxt & ~(iuMask | ivMask)) | iuNxt | (ivNxt << ivShift);
      if (partialWafer) {
        if (hgc_->waferExist(idNxt.layer(), iuNxt, ivNxt))
          icount++;
        else
          detIdVec[icount] = 0;
      } else
        icount++;
    }
  }

  return detIdVec;
}
