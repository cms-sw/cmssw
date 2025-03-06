#ifndef Phase2L1Trigger_DTTrigger_ShowerBuilder_h
#define Phase2L1Trigger_DTTrigger_ShowerBuilder_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "L1Trigger/DTTriggerPhase2/interface/ShowerCandidate.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include <iostream>
#include <fstream>

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

namespace showerb {
    typedef std::pair<int, DTPrimitive> DTPrimPlusBx;
    typedef std::deque<DTPrimPlusBx> ShowerBuffer;
    
    bool hitWireSort_shower(const DTPrimitive& hit1, const DTPrimitive& hit2){
        int wi1 = hit1.channelId();
        int wi2 = hit2.channelId();

        if (wi1 < wi2) return true;
        else return false;
    }

    bool hitLayerSort_shower(const DTPrimitive& hit1, const DTPrimitive& hit2) {
        int lay1 = hit1.layerId();
        int lay2 = hit2.layerId();

        if (lay1 < lay2) return true;
        else if (lay1 > lay2) return false;
        else return hitWireSort_shower(hit1, hit2);
    }

    bool hitTimeSort_shower(const DTPrimitive& hit1, const DTPrimitive& hit2) {
        int tdc1 = hit1.tdcTimeStamp();
        int tdc2 = hit2.tdcTimeStamp();

        if (tdc1 < tdc2) return true;
        else return false;
        // else if (tdc1 > tdc2) return false;
        // else return hitLayerSort_shower(hit1, hit2); --> ignoring those sortings for now
    }

    float compute_avg_pos(DTPrimitives& hits) { 
        int nhits_ = hits.size();
        
        if (nhits_ == 0) return -1.0;
        
        float aux_avgPos_ = 0;
        
        for (auto& hit : hits) {
            aux_avgPos_ += hit.wireHorizPos();
        }
        
        return aux_avgPos_ / nhits_;
    }
    
    float compute_avg_time(DTPrimitives& hits) {
        int nhits_ = hits.size();

        if (nhits_ == 0) return -1.0;

        float aux_avgTime_ = 0;
        
        for (auto& hit : hits) {
            aux_avgTime_ += hit.tdcTimeStamp();
        }
        return aux_avgTime_ / nhits_;
    }

    void set_wires_profile(std::vector<int>& wires_profile, DTPrimitives& hits) {
        for (auto& hit : hits) {
            wires_profile[hit.channelId()-1]++;
        }
    }

    bool buffer_contains(const ShowerBuffer& buffer, const DTPrimitive& hit) {
        for (const auto& item : buffer) {
            if (item.second.channelId() == hit.channelId()) return true;
        }
        return false;
    }

    void buffer_get_hits(const ShowerBuffer& buffer, DTPrimitives& hits) {
        for (const auto& item : buffer) {
            hits.push_back(item.second);
        }
    }

    void buffer_clear_olds(ShowerBuffer& buffer, const int _current_bx, const int persistence_bx_units) {
        while (!buffer.empty() && (_current_bx - buffer.front().first) > persistence_bx_units) {
            buffer.pop_front();
        }
    }

    void buffer_reset(ShowerBuffer& buffer) {
        buffer.clear();
    }
}

// ===============================================================================
// Class declarations
// ===============================================================================

class ShowerBuilder {

public:
    // Constructors and destructor
    ShowerBuilder(const edm::ParameterSet& pset, edm::ConsumesCollector& iC);
    virtual ~ShowerBuilder();

    // Main methods
    virtual void initialise(const edm::EventSetup& iEventSetup);
    virtual void run(edm::Event& iEvent,
                     const edm::EventSetup& iEventSetup,
                     const DTDigiCollection& digis,
                     ShowerCandidatePtr &showerCandidate_SL1,
                     ShowerCandidatePtr &showerCandidate_SL3);
    virtual void finish();

private:
    // Private auxiliary methods
    void clear();
    void setInChannels(const DTDigiCollection* digi);
    void processHits_standAlone(std::map<int, ShowerCandidatePtr> &showerCands);
    void processHitsFirmwareEmulation(std::map<int, ShowerCandidatePtr> &showerCands);

    bool triggerShower(const showerb::ShowerBuffer& buffer);
    void set_shower_properties(
        ShowerCandidatePtr &showerCand,
        showerb::ShowerBuffer& buffer, 
        int nhits = -1,
        int bx = -1,
        int min_wire = -1,
        int max_wire = -1,
        float avg_pos = -1.,
        float avg_time = -1.
    );
    void groupHits_byBx();
    void fill_obdt(const int bx);
    void fill_bmtl1_buffers();
    void bxStep(const int _current_bx);

    // Private attributes
    const int showerTaggingAlgo_;
    const int threshold_for_shower_;
    const int nHits_per_bx_;
    const int obdt_hits_bxpersistence_;
    const int obdt_wire_relaxing_time_;
    const int bmtl1_hits_bxpersistence_;
    const bool debug_;
    const int scenario_;

    // auxiliary variables
    DTPrimitives all_hits;
    std::map<int, DTPrimitives, std::less<int>> all_hits_perBx;
    showerb::ShowerBuffer obdt_buffer; // Buffer to emulate the OBDT behavior
    showerb::ShowerBuffer hot_wires_buffer; // Buffer to emulate the hot wires behavior
    showerb::ShowerBuffer bmtl1_sl1_buffer; // Buffer to emulate the BMTL1 shower buffer for SL1
    showerb::ShowerBuffer bmtl1_sl3_buffer; // Buffer to emulate the BMTL1 shower buffer for SL3
};

#endif
