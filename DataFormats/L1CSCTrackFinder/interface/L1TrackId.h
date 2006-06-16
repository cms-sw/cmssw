/**
 * \class L1TrackId
 * \author D. Acosta (chages by L. Gray)
 *
 * A wrapper for CSCDetId so that it can label a track.
 * Also tools to help determine what LCTs make up the track.
 */

#ifndef L1CSCTrackFinder_L1TrackId_h
#define L1CSCTrackFinder_L1TrackId_h

#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>

namespace csc 
{

  class L1TrackId
    {
    public:
      enum { kRankBitWidth = 6 };

      L1TrackId();
      L1TrackId(const unsigned& side, const unsigned& sector);
      L1TrackId(const csc::L1TrackId&);

      const L1TrackId& operator=(const csc::L1TrackId&);

      inline unsigned endcap() const { return id_.endcap(); }
      inline unsigned sector() const { return CSCTriggerNumbering::triggerSectorFromLabels(id_); }
      inline unsigned station() const { return id_.station(); }
      
      inline unsigned rank() const { return m_rank; }
      inline unsigned mode() const { return m_mode; }
      inline unsigned numSegments() const { return 0; } // finish later
    
      void setRank(const unsigned& rank);
      void setMode(const unsigned& mode) { m_mode = mode; } 
      void setOverlap(const unsigned& rank);

      bool sharesHit(const csc::L1TrackId&) const;
      inline bool inOverlap() const { return m_overlap; }
      
      void overlapMode(const unsigned& rank, int& mode, int& stnA, int& stnB);
      unsigned encodeLUTMode(const unsigned& rank) const;

      //void addSegment(const CSCDetId& id, const CSCCorrelatedLCTDigi& digi) { stubs.insertDigi(id,digi); }

      /// Functions to determine which stations are in this track.
      bool hasME1(const unsigned& rank) const;
      bool hasME2(const unsigned& rank) const;
      bool hasME3(const unsigned& rank) const;
      bool hasME4(const unsigned& rank) const;
      bool hasMB1(const unsigned& rank) const; 

      inline bool erase() const { return m_erase; }

    private:
      unsigned m_rank, m_mode;
      bool m_erase, m_overlap;

      CSCDetId id_;

      //CSCCorrelatedLCTDigiCollection stubs;
    };

}

#endif
