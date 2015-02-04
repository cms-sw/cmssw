#ifndef DSpappa_HH
#define DSpappa_HH

//
// Federico Ferri, CEA-Saclay Irfu/SPP, 14.12.2011
// federico.ferri@cern.ch
//

#include <cassert>
#include "Calibration/Tools/interface/IC.h"

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


class DSIsBarrelPlus : public DS
{
        public:
                bool operator()(DetId id)
                {
                        if (id.subdetId() == EcalBarrel)
                        {
                                return EBDetId(id).ieta() > 0;
                        }
                        return false;
                }
};


class DSIsBarrelMinus : public DS
{
        public:
                bool operator()(DetId id)
                {
                        if (id.subdetId() == EcalBarrel)
                        {
                                return EBDetId(id).ieta() < 0;
                        }
                        return false;
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


class DSIsUncalib : public DS
{
        public:
                DSIsUncalib() : set_(false) { }
                void setIC(const IC & ic) { ic_ = ic; set_ = true; }
                bool operator()(DetId id)
                {
                        assert(set_);
                        float v = ic_.ic()[id];
                        float e = ic_.eic()[id];
                        return !IC::isValid(v, e);
                }
        private:
                IC ic_;
                bool set_;
};


class DSIsOutlier : public DS
{
        public:
                DSIsOutlier() : set_ic_(false), set_thr_(false) { }
                void setIC(const IC & ic) { ic_ = ic; set_ic_ = true; }
                void setThresholds(float min, float max, bool err = false) { min_ = min; max_ = max; err_ = err; set_thr_ = true; }
                bool operator()(DetId id)
                {
                        assert(set_ic_ && set_thr_);
                        float v;
                        if (err_) v = ic_.eic()[id];
                        else      v = ic_.ic()[id];
                        return (v < min_ || v > max_);
                }
        private:
                IC ic_;
                float min_;
                float max_;
                bool err_;
                bool set_ic_;
                bool set_thr_;
};

#endif
