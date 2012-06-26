#ifndef IC_HH
#define IC_HH

//
// Federico Ferri, CEA-Saclay Irfu/SPP, 14.12.2011
// federico.ferri@cern.ch
//

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibErrors.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TRandom.h"
#include "TTree.h"

#include "Calibration/Tools/interface/DRings.h"

class DS
{
        public:
                DS() : cnt_(0) {}
                virtual bool operator()(DetId id) = 0;

                int cnt() { return cnt_; }
                void reset() { cnt_ = 0; }
        protected:
                int cnt_;
};

class DSAll : public DS
{
        public:
                bool operator()(DetId id)
                {
                        return true;
                }
};

class DSIsBorderNeighbour : public DS
{
        public:
                bool operator()(DetId id)
                {
                        return true;
                }
};


class DSIsDeadNeighbour : public DS
{
        public:
                bool operator()(DetId id)
                {
                        return true;
                }
};



class DSIsBarrel : public DS
{
        public:
                bool operator()(DetId id)
                {
                        return id.subdetId() == EcalBarrel;
                }
};


class DSIsEndcap : public DS
{
        public:
                bool operator()(DetId id)
                {
                        return id.subdetId() == EcalEndcap;
                }
};


class DSIsEndcapPlus : public DS
{
        public:
                bool operator()(DetId id)
                {
                        if (id.subdetId() == EcalEndcap) {
                                return EEDetId(id).zside() > 0;
                        }
                        return false;
                }
};


class DSIsEndcapMinus : public DS
{
        public:
                bool operator()(DetId id)
                {
                        if (id.subdetId() == EcalEndcap) {
                                return EEDetId(id).zside() < 0;
                        }
                        return false;
                }
};


class DSIsNextToBoundaryEB : public DS
{
        public:
                bool operator()(DetId id)
                {
                        if (id.subdetId() == EcalBarrel) {
                                return EBDetId::isNextToBoundary(id);
                        }
                        return false;
                }
};

class DSIsNextToProblematicEB : public DS
{
        public:
                DSIsNextToProblematicEB() : set_(false), thr_(6) {};
                void setChannelStatus(EcalChannelStatus & chStatus) { channelStatus_ = chStatus; set_ = true; }
                void setStatusThreshold(int thr) { thr_ = thr; }
                int statusThreshold() { return thr_; }
                bool operator()(DetId id)
                {
                        assert(set_);
                        if (id.subdetId() != EcalBarrel) return false;
                        // exclude the dead themselves
                        if (channelStatus_.find(id)->getStatusCode() > thr_) return false;
                        for (int i = -1; i <= 1; ++i) {
                                for (int j = -1; j <= 1; ++j) {
                                        if (i != 0 || j != 0) {
                                                DetId tid = EBDetId::offsetBy(id, i, j);
                                                if (tid != DetId(0) && channelStatus_.find(tid)->getStatusCode() > thr_) {
                                                        ++cnt_;
                                                        return true;
                                                }
                                        }
                                }
                        }
                        return false;
                }

        private:
                EcalChannelStatus channelStatus_;
                bool set_;
                int  thr_;
};


class DSHasChannelStatusEB : public DS
{
        public:
                DSHasChannelStatusEB() : set_(false), val_(0) {};
                void setChannelStatus(EcalChannelStatus & chStatus) { channelStatus_ = chStatus; set_ = true; }
                void setStatusValue(int val) { val_ = val; }
                int statusValue() { return val_; }
                bool operator()(DetId id)
                {
                        assert(set_);
                        if (id.subdetId() != EcalBarrel) return false;
                        // exclude the dead themselves
                        if (channelStatus_.find(id)->getStatusCode() != val_) return false;
                        return true;
                }

        private:
                EcalChannelStatus channelStatus_;
                bool set_;
                int  val_;
};


class DSHasChannelStatusEE : public DS
{
        public:
                DSHasChannelStatusEE() : set_(false), val_(0) {};
                void setChannelStatus(EcalChannelStatus & chStatus) { channelStatus_ = chStatus; set_ = true; }
                void setStatusValue(int val) { val_ = val; }
                int statusValue() { return val_; }
                bool operator()(DetId id)
                {
                        assert(set_);
                        if (id.subdetId() != EcalEndcap) return false;
                        // exclude the dead themselves
                        if (channelStatus_.find(id)->getStatusCode() != val_) return false;
                        return true;
                }

        private:
                EcalChannelStatus channelStatus_;
                bool set_;
                int  val_;
};


class DSIsNextToNextToProblematicEB : public DS
{
        public:
                DSIsNextToNextToProblematicEB() : set_(false), thr_(6) { isNextToProblematicEB_.setChannelStatus(channelStatus_); };
                void setChannelStatus(EcalChannelStatus & chStatus) { channelStatus_ = chStatus; isNextToProblematicEB_.setChannelStatus(channelStatus_); set_ = true; }
                void setStatusThreshold(int thr) { thr_ = thr; }
                int statusThreshold() { return thr_; }
                bool operator()(DetId id)
                {
                        assert(set_);
                        if (id.subdetId() != EcalBarrel) return false;
                        // exclude the dead themselves
                        if (channelStatus_.find(id)->getStatusCode() > thr_) return false;
                        for (int i = -2; i <= 2; ++i) {
                                for (int j = -2; j <= 2; ++j) {
                                        //if ((i != 0 || j != 0) && abs(i) != 1 && abs(j) != 1) {
                                        if (abs(i) == 2 || abs(j) == 2) {
                                                DetId tid = EBDetId::offsetBy(id, i, j);
                                                if (tid != DetId(0) && channelStatus_.find(tid)->getStatusCode() > thr_) {
                                                        ++cnt_;
                                                        return !isNextToProblematicEB_(id);
                                                }
                                        }
                                }
                        }
                        return false;
                }

        private:
                EcalChannelStatus channelStatus_;
                DSIsNextToProblematicEB isNextToProblematicEB_;
                bool set_;
                int  thr_;
};


class DSIsNextToProblematicEE : public DS
{
        public:
                DSIsNextToProblematicEE() : set_(false), thr_(6) {}
                void setChannelStatus(EcalChannelStatus & chStatus) { channelStatus_ = chStatus; set_ = true; }
                void setStatusThreshold(int thr) { thr_ = thr; }
                int statusThreshold() { return thr_; }
                bool operator()(DetId id)
                {
                        assert(set_);
                        if (id.subdetId() != EcalEndcap) return false;
                        // exclude the dead themselves
                        if (channelStatus_.find(id)->getStatusCode() > thr_) return false;
                        for (int i = -1; i < 2; ++i) {
                                for (int j = -1; j < 2; ++j) {
                                        if (i != 0 || j != 0) {
                                                DetId tid = EEDetId::offsetBy(id, i, j);
                                                if (tid != DetId(0) && channelStatus_.find(tid)->getStatusCode() > thr_) {
                                                        ++cnt_;
                                                        return true;
                                                }
                                        }
                                }
                        }
                        return false;
                }
        private:
                EcalChannelStatus channelStatus_;
                bool set_;
                int  thr_;
};


class DSIsNextToProblematicEEPlus : public DS
{
        public:
                DSIsNextToProblematicEEPlus() : set_(false) {}
                void setChannelStatus(EcalChannelStatus & channelStatus) { isNextToProblematicEE.setChannelStatus(channelStatus); set_ = true; };
                bool operator()(DetId id)
                {
                        assert(set_);
                        if (id.subdetId() != EcalEndcap) return false;
                        return isNextToProblematicEE(id) && (EEDetId(id).zside() > 0);
                }
        private:
                DSIsNextToProblematicEE isNextToProblematicEE;
                bool set_;
};


class DSIsNextToProblematicEEMinus : public DS
{
        public:
                DSIsNextToProblematicEEMinus() : set_(false) {}
                void setChannelStatus(EcalChannelStatus & channelStatus) { isNextToProblematicEE.setChannelStatus(channelStatus); set_ = true; };
                bool operator()(DetId id)
                {
                        assert(set_);
                        if (id.subdetId() != EcalEndcap) return false;
                        return isNextToProblematicEE(id) && (EEDetId(id).zside() < 0);
                }
        private:
                DSIsNextToProblematicEE isNextToProblematicEE;
                bool set_;
};


class DSRandom : public DS
{
        public:
                DSRandom() : frac_(0) { r = new TRandom(); }
                DSRandom(float frac) : frac_(frac) { r = new TRandom(); }
                ~DSRandom() { if (r) delete r; }
                void setFraction(float frac) { frac_ = frac; }
                void setSeed(UInt_t seed) { r->SetSeed(seed); }
                UInt_t seed() { return r->GetSeed(); }
                float fraction() { return frac_; }
                bool operator()(DetId id)
                {
                        if (id.subdetId() != EcalBarrel) return false;
                        if (r->Uniform() < frac_) {
                                ++cnt_;
                                return true;
                        } else  return false;
                }
        private:
                float frac_;
                TRandom * r;
};


class IC {
        public:
                enum EcalPart { kAll, kEB, kEE };

                IC();

                EcalIntercalibConstants & ic() { return _ic; }
                EcalIntercalibErrors & eic() { return _eic; }

                const EcalIntercalibConstants & ic() const { return _ic; }
                const EcalIntercalibErrors & eic() const { return _eic; }
                const std::vector<DetId> & ids() const { return _detId; }
                void setRings(const DRings & dr) { dr_ = dr; idr_ = true; }

                // plotters
                static void constantMap(const IC & a, TH2F * h, DS & d, bool errors = false);
                static void constantDistribution(const IC & a, TH1F * h, DS & d, bool errors = false);
                static void profileEta(const IC & a, TProfile * h, DS & d, bool errors = false);
                static void profilePhi(const IC & a, TProfile * h, DS & d, bool errors = false);
                static void profileSM(const IC & a, TProfile * h, DS & d, bool errors = false);

                static bool isValid(float v, float e);

                // IC manipulation
                static void reciprocal(const IC & a, IC & res);
                static void multiply(const IC & a, float c, IC & res, DS & d);
                static void multiply(const IC & a, const IC & b, IC & res);
                static void add(const IC & a, const IC & b, IC & res);
                static void combine(const IC & a, const IC & b, IC & res);
                static void fillHoles(const IC & a, const IC & b, IC & res);
                static void smear(const IC & a, float sigma, IC & res);
                static void smear(const IC & a, IC & res);

                // tools
                static void applyEtaScale(IC & ic);
                static void scaleEta(IC & ic, const IC & ic_scale, bool reciprocalScale = false);
                static void applyTwoCrystalEffect(IC & ic);
                static void setToUnit(IC & ic);
                static void dump(const IC & a, const char * fileName, DS & d);
                static void dumpXML(const IC & a, const char * fileName, DS & d, bool errors = false);
                static void readSimpleTextFile(const char * fileName, IC & ic);
                static void readTextFile(const char * fileName, IC & ic);
                static void readXMLFile(const char * fileName, IC & ic);
                static void readCmscondXMLFile(const char * fileName, IC & ic);
                static void readEcalChannelStatusFromTextFile(const char * fileName, EcalChannelStatus & channelStatus);
                static void makeRootTree(TTree & t, const IC & ic);

                static float average(const IC & a, DS & d, bool errors = false);

        private:
                EcalIntercalibConstants   _ic;
                EcalIntercalibErrors     _eic;
                static DRings dr_;
                static bool idr_;
                static std::vector<DetId>     _detId;
};

#endif
