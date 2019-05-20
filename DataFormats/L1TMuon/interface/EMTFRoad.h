#ifndef DataFormats_L1TMuon_EMTFRoad_h
#define DataFormats_L1TMuon_EMTFRoad_h

#include <cstdint>
#include <vector>

namespace l1t {

  class EMTFRoad {
  public:
    EMTFRoad()
        : endcap(-99),
          sector(-99),
          sector_idx(-99),
          bx(-99),
          zone(-99),
          key_zhit(-99),
          pattern(-99),
          straightness(-99),
          layer_code(-99),
          quality_code(-99),
          winner(-99){};

    virtual ~EMTFRoad(){};

    void set_endcap(int bits) { endcap = bits; }
    void set_sector(int bits) { sector = bits; }
    void set_sector_idx(int bits) { sector_idx = bits; }
    void set_bx(int bits) { bx = bits; }
    void set_zone(int bits) { zone = bits; }
    void set_key_zhit(int bits) { key_zhit = bits; }
    void set_pattern(int bits) { pattern = bits; }
    void set_straightness(int bits) { straightness = bits; }
    void set_layer_code(int bits) { layer_code = bits; }
    void set_quality_code(int bits) { quality_code = bits; }
    void set_winner(int bits) { winner = bits; }

    int Endcap() const { return endcap; }
    int Sector() const { return sector; }
    int Sector_idx() const { return sector_idx; }
    int BX() const { return bx; }
    int Zone() const { return zone; }
    int Key_zhit() const { return key_zhit; }
    int Pattern() const { return pattern; }
    int Straightness() const { return straightness; }
    int Layer_code() const { return layer_code; }
    int Quality_code() const { return quality_code; }
    int Winner() const { return winner; }

  private:
    int endcap;
    int sector;
    int sector_idx;
    int bx;
    int zone;      // Pattern detector ID
    int key_zhit;  // Also called 'ph_num' or 'ph_pat'
    int pattern;   // Pattern detector ID
    int straightness;
    int layer_code;
    int quality_code;  // Used to be 'rank'.  Also called 'ph_q'
    int winner;        // 0 is first winner, 1 is second, etc.

  };  // End of class EMTFRoad

  // Define a vector of EMTFRoad
  typedef std::vector<EMTFRoad> EMTFRoadCollection;

}  // End of namespace l1t

#endif /* define DataFormats_L1TMuon_EMTFRoad_h */
