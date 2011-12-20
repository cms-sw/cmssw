#include "Calibration/Tools/interface/IC.h"

std::vector<DetId> IC::_detId;
EcalChannelStatus IC::channelStatus;

IC::IC()
{
        _detId.resize(EBDetId::MAX_HASH + 1 + EEDetId::kSizeForDenseIndexing);
        int idx = -1;
        for (int hi = EBDetId::MIN_HASH; hi <= EBDetId::MAX_HASH; ++hi ) {
                EBDetId ebId = EBDetId::unhashIndex(hi);
                if (ebId != EBDetId()) {
                        _detId[hi] = ebId;
                }
        }
        for ( int hi = 0; hi < EEDetId::kSizeForDenseIndexing; ++hi ) {
                EEDetId eeId = EEDetId::unhashIndex(hi);
                if (eeId != EEDetId()) {
                        idx = EBDetId::MAX_HASH + 1 + hi;
                        _detId[idx] = eeId;
                }
        }
        assert(_detId.size() == 75848);
        assert(_detId.size() == EBDetId::MAX_HASH + 1 + EEDetId::kSizeForDenseIndexing);
}


bool IC::all(DetId id)
{
        return true;
}

bool IC::isBorderNeighbour(DetId id)
{
        return true;
}

//enum DeadKind = { kSingle, kVFE, kTower };

bool IC::isDeadNeighbour(DetId id, EcalChannelStatus & chStatus)
{
        return true;
}


bool IC::isBarrel(DetId id)
{
        return id.subdetId() == EcalBarrel;
}


bool IC::isEndcap(DetId id)
{
        return id.subdetId() == EcalEndcap;
}


bool IC::isEndcapPlus(DetId id)
{
        if (id.subdetId() == EcalEndcap) {
                return EEDetId(id).zside() > 0;
        }
        return false;
}


bool IC::isEndcapMinus(DetId id)
{
        if (id.subdetId() == EcalEndcap) {
                return EEDetId(id).zside() < 0;
        }
        return false;
}


bool IC::isNextToBoundaryEB(DetId id)
{
        if (id.subdetId() == EcalBarrel) {
                return EBDetId::isNextToBoundary(id);
        }
        return false;
}


bool IC::isNextToProblematicEB(DetId id)
{
        if (id.subdetId() != EcalBarrel) return false;
        // exclude the dead themselves
        if (channelStatus.find(id)->getStatusCode() > 6) return false;
        for (int i = -1; i < 2; ++i) {
                for (int j = -1; j < 2; ++j) {
                        if (i != 0 || j != 0) {
                                DetId tid = EBDetId::offsetBy(id, i, j);
                                if (tid != DetId(0) && channelStatus.find(tid)->getStatusCode() > 6) return true;
                        }
                }
        }
        return false;
}


bool IC::isNextToProblematicEE(DetId id)
{
        if (id.subdetId() != EcalEndcap) return false;
        // exclude the dead themselves
        if (channelStatus.find(id)->getStatusCode() > 6) return false;
        for (int i = -1; i < 2; ++i) {
                for (int j = -1; j < 2; ++j) {
                        if (i != 0 || j != 0) {
                                DetId tid = EEDetId::offsetBy(id, i, j);
                                if (tid != DetId(0) && channelStatus.find(tid)->getStatusCode() > 6) return true;
                        }
                }
        }
        return false;
}

bool IC::isNextToProblematicEEPlus(DetId id)
{
        if (id.subdetId() != EcalEndcap) return false;
        return isNextToProblematicEE(id) && (EEDetId(id).zside() > 0);
}

bool IC::isNextToProblematicEEMinus(DetId id)
{
        if (id.subdetId() != EcalEndcap) return false;
        return isNextToProblematicEE(id) && (EEDetId(id).zside() < 0);
}

//bool isNextToBoundaryEE(DetId id)
//{
//        if (id.subdetId() == EcalBarrel) {
//                return EBDetId::isNextToBoundary(id);
//        } else if (id.subdetId() == EcalEndcap) {
//                return EEDetId::isNextToBoundary(id);
//        }
//        return false;
//}


void IC::constantDistribution(const IC & a, TH1F * h, bool (*selector)(DetId id))
{
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                if (!selector(id)) continue;
                h->Fill(a.ic()[id]);
        }
}


void IC::constantMap(const IC & a, TH2F * h, bool (*selector)(DetId id))
{
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                if (!selector(id)) continue;
                int ix = 0, iy = 0, iz = 0;
                if (id.subdetId() == EcalBarrel) {
                        ix = EBDetId(id).iphi();
                        iy = EBDetId(id).ieta();
                        iz = 0;
                } else if (id.subdetId() == EcalEndcap) {
                        ix = EEDetId(id).ix();
                        iy = EEDetId(id).iy();
                        iz = EEDetId(id).zside();
                }
                h->Fill(ix, iy, a.ic()[id]);
        }
}


void IC::profileEta(const IC & a, TProfile * h, bool (*selector)(DetId id))
{
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                if (!selector(id)) continue;
                int x = 0;
                if (id.subdetId() == EcalBarrel) {
                        x = EBDetId(id).ieta();
                } else if (id.subdetId() == EcalEndcap) {
                        x = 100;
                }
                h->Fill(x, a.ic()[id]);
        }
}


void IC::profilePhi(const IC & a, TProfile * h, bool (*selector)(DetId id))
{
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                if (!selector(id)) continue;
                int x = 0;
                if (id.subdetId() == EcalBarrel) {
                        x = EBDetId(id).iphi();
                } else if (id.subdetId() == EcalEndcap) {
                        //x = atan((EEDetId(id).iy() - 50) / (EEDetId(id).ix() - 50));
                }
                h->Fill(x, a.ic()[id]);
        }
}


void IC::profileSM(const IC & a, TProfile * h, bool (*selector)(DetId id))
{
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                if (!selector(id) || !isBarrel(id)) continue;
                h->Fill(EBDetId(id).ism(), a.ic()[id]);
        }
}


double IC::average(const IC & a, bool (*selector)(DetId id))
{
        double ave = 0.;
        int cnt = 0;
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                if (!selector(id)) continue;
                ++cnt;
                ave += a.ic()[id];
        }
        return ave / cnt;
}


void IC::reciprocal(const IC & a, IC & res)
{
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                float v = a.ic()[id];
                float e = a.eic()[id];
                res.ic().setValue(id, 1. / v);
                res.eic().setValue(id, v / e / e);
        }
}


void IC::multiply(const IC & a, float c, IC & res)
{
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                res.ic().setValue(id, a.ic()[id] * c);
                res.eic().setValue(id, a.eic()[id] * fabs(c));
        }
}


void IC::multiply(const IC & a, const IC & b, IC & res)
{
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                float va = a.ic()[id];
                float ea = a.eic()[id];
                float vb = b.ic()[id];
                float eb = b.eic()[id];
                res.ic().setValue(id, va * vb);
                res.eic().setValue(id, sqrt(vb * vb * ea * ea + va * va * eb * eb));
        }
}


void IC::add(const IC & a, const IC & b, IC & res)
{
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                float va = a.ic()[id];
                float ea = a.eic()[id];
                float vb = b.ic()[id];
                float eb = b.eic()[id];
                res.ic().setValue(id, va + vb);
                res.eic().setValue(id, sqrt(ea * ea + eb * eb));
        }
}


void IC::combine(const IC & a, const IC & b, IC & res)
{
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                float va = a.ic()[id];
                float wa = 1. / a.eic()[id] / a.eic()[id];
                float vb = b.ic()[id];
                float wb = 1. / b.eic()[id] / b.eic()[id];
                res.ic().setValue(id, (wa * va + wb * vb) / (wa + wb));
                res.eic().setValue(id, 1. / sqrt(wa + wb));
        }
}


void IC::smear(const IC & a, float sigma, IC & res)
{
        TRandom * r = new TRandom();
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                float s = r->Gaus(1, sigma);
                res.ic().setValue(id, a.ic()[id] * s);
                res.eic().setValue(id, a.eic()[id] * s);
        }
        delete r;
}


void IC::dump(const IC & a, const char * fileName, bool (*selector)(DetId id))
{
        FILE * fd = fopen(fileName, "w");
        if (fd == NULL) {
                fprintf(stderr, "[dump] cannot open file %s\n", fileName);
                exit(-1);
        }
        for (size_t i = 0; i < a.ids().size(); ++i) {
                DetId id(a.ids()[i]);
                if (!selector(id)) continue;
                fprintf(fd, "%d %f %f\n", id.rawId(), a.ic()[id], a.eic()[id]);
        }
        fclose(fd);
}


void IC::readSimpleTextFile(const char * fileName, IC & ic)
{
        FILE * fd = fopen(fileName, "r");
        if (fd == NULL) {
                fprintf(stderr, "[readSimpleTextFile] cannot open file %s\n", fileName);
                exit(-1);
        }
        char * line = NULL;
        size_t len = 0;
        ssize_t read;
        int id;
        float c, e;
        while ((read = getline(&line, &len, fd)) != EOF) {
                if (line[0] == '#') continue;
                sscanf(line, "%d %f %f", &id, &c, &e);
                ic.ic().setValue(id, c);
                ic.eic().setValue(id, e);
        }
        fclose(fd);
}


//void IC::readEcalChannelStatusFromTextFile(const char * fileName, EcalChannelStatus & ch)
void IC::readEcalChannelStatusFromTextFile(const char * fileName)
{
        FILE * fd = fopen(fileName, "r");
        if (fd == NULL) {
                fprintf(stderr, "[readSimpleTextFile] cannot open file %s\n", fileName);
                exit(-1);
        }
        uint32_t id;
        uint16_t st;
        int cnt = 0;
        while (fscanf(fd, "%u %hu", &id, &st) != EOF) {
                EcalChannelStatusCode code(st);
                channelStatus[id] = code;
                ++cnt;
        }
        assert(cnt == 75848);
        fclose(fd);
}


void IC::makeRootTree(TTree & t, const IC & ic)
{
}
