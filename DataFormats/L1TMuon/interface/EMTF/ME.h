// Class for Muon Endcap (ME) Data Record

#ifndef __l1t_emtf_ME_h__
#define __l1t_emtf_ME_h__

#include <boost/cstdint.hpp> 

namespace l1t {
  namespace emtf {
    class ME {
    public:
      
      explicit ME(uint64_t dataword);
    
    ME() : 
      me_bxn(-99), wire(-99), strip(-99), quality(-99), 
	clct_pattern(-99), csc_ID(-99), epc(-99), station(-99), neighbor(-99), sector(-99),
	subsector(-99), ring(-99), tbin_num(-99), bc0(-99), bxe(-99), lr(-99), afff(-99), 
	cik(-99), nit(-99), afef(-99), se(-99), sm(-99), af(-99), vp(-99), stub_num(-99), 
	format_errors(0), dataword(-99) 
	{};
      
      virtual ~ME() {};
      
      // Converts station, CSC_ID, sector, subsector, and neighbor
      std::vector<int> convert_chamber_ME(int _station, int _csc_ID, int _sector) {
        int new_sector = _sector;
	int new_csc_ID = _csc_ID;
	if      (_station == 0) { int arr[] = {       1, new_csc_ID, new_sector,   1, 0}; std::vector<int> vec(arr, arr+5); return vec; }
	else if (_station == 1) { int arr[] = {       1, new_csc_ID, new_sector,   2, 0}; std::vector<int> vec(arr, arr+5); return vec; }
        else if (_station <= 4) { int arr[] = {_station, new_csc_ID, new_sector, -99, 0}; std::vector<int> vec(arr, arr+5); return vec; }
        else if (_station == 5) new_sector = (_sector != 1) ? _sector-1 : 6;
	else { int arr[] = {-99, new_csc_ID, _sector, -99, -99}; std::vector<int> vec(arr, arr+5); return vec; }

        if      (new_csc_ID == 1) { int arr[] = {1, 3, new_sector,   2, 1}; std::vector<int> vec(arr, arr+5); return vec; }
        else if (new_csc_ID == 2) { int arr[] = {1, 6, new_sector,   2, 1}; std::vector<int> vec(arr, arr+5); return vec; }
        else if (new_csc_ID == 3) { int arr[] = {1, 9, new_sector,   2, 1}; std::vector<int> vec(arr, arr+5); return vec; }
        else if (new_csc_ID == 4) { int arr[] = {2, 3, new_sector, -99, 1}; std::vector<int> vec(arr, arr+5); return vec; }
        else if (new_csc_ID == 5) { int arr[] = {2, 9, new_sector, -99, 1}; std::vector<int> vec(arr, arr+5); return vec; }
        else if (new_csc_ID == 6) { int arr[] = {3, 3, new_sector, -99, 1}; std::vector<int> vec(arr, arr+5); return vec; }
        else if (new_csc_ID == 7) { int arr[] = {3, 9, new_sector, -99, 1}; std::vector<int> vec(arr, arr+5); return vec; }
        else if (new_csc_ID == 8) { int arr[] = {4, 3, new_sector, -99, 1}; std::vector<int> vec(arr, arr+5); return vec; }
        else if (new_csc_ID == 9) { int arr[] = {4, 9, new_sector, -99, 1}; std::vector<int> vec(arr, arr+5); return vec; }
        else                   { int arr[] = {5, new_csc_ID, new_sector, -99, -99}; std::vector<int> vec(arr, arr+5); return vec; }
      }

      // Calculates ring value
      int calc_ring_ME(int _station, int _csc_ID, int _strip) {
	if (_station > 1) {
	  if      (_csc_ID <  4) return 1;
	  else if (_csc_ID < 10) return 2;
	  else return -99;
	}
	else if (_station == 1) {
	  if      (_csc_ID < 4 && _strip > 127) return 4;
	  else if (_csc_ID < 4 && _strip >=  0) return 1;
	  else if (_csc_ID > 3 && _csc_ID <  7) return 2;
	  else if (_csc_ID > 6 && _csc_ID < 10) return 3;
	  else return -99;
	}
	else return -99;
      }
      
      void set_me_bxn(int bits)              { me_bxn = bits;              };
      void set_wire(int bits)                { wire = bits;                };
      void set_strip(int bits)               { strip = bits;               };
      void set_quality(int bits)             { quality = bits;             };
      void set_clct_pattern(int bits)        { clct_pattern = bits;        };
      void set_csc_ID(int bits)              { csc_ID = bits;              };
      void set_epc(int bits)                 { epc = bits;                 };
      void set_station(int bits)             { station = bits;             };
      void set_neighbor(int bits)            { neighbor = bits;            };
      void set_sector(int bits)              { sector = bits;              };
      void set_subsector(int bits)           { subsector = bits;           };
      void set_ring(int bits)                { ring = bits;                };
      void set_tbin_num(int bits)            { tbin_num = bits;            };
      void set_bc0(int bits)                 { bc0 = bits;                 };
      void set_bxe(int bits)                 { bxe = bits;                 };
      void set_lr(int bits)                  { lr = bits;                  };
      void set_afff(int bits)                { afff = bits;                };
      void set_cik(int bits)                 { cik = bits;                 };
      void set_nit(int bits)                 { nit = bits;                 };
      void set_afef(int bits)                { afef = bits;                };
      void set_se(int bits)                  { se = bits;                  };
      void set_sm(int bits)                  { sm = bits;                  };
      void set_af(int bits)                  { af = bits;                  };
      void set_vp(int bits)                  { vp = bits;                  };
      void set_stub_num(int bits)            { stub_num = bits;            };
      void add_format_error()                { format_errors += 1;         };
      void set_dataword(uint64_t bits)       { dataword = bits;            };

      int      ME_BXN()              const { return me_bxn;              };
      int      Wire()                const { return wire;                };
      int      Strip()               const { return strip;               };
      int      Quality()             const { return quality;             };
      int      CLCT_pattern()        const { return clct_pattern;        };
      int      CSC_ID()              const { return csc_ID;              };
      int      EPC()                 const { return epc;                 };
      int      Station()             const { return station;             };
      int      Neighbor()            const { return neighbor;            };
      int      Sector()              const { return sector;              };
      int      Subsector()           const { return subsector;           };
      int      Ring()                const { return ring;                };
      int      Tbin_num()            const { return tbin_num;            };
      int      BC0()                 const { return bc0;                 };
      int      BXE()                 const { return bxe;                 };
      int      LR()                  const { return lr;                  };
      int      AFFF()                const { return afff;                };
      int      CIK()                 const { return cik;                 };
      int      NIT()                 const { return nit;                 };
      int      AFEF()                const { return afef;                };
      int      SE()                  const { return se;                  };
      int      SM()                  const { return sm;                  };
      int      AF()                  const { return af;                  };
      int      VP()                  const { return vp;                  };      
      int      Stub_num()            const { return stub_num;            };      
      int      Format_Errors()       const { return format_errors;        };
      uint64_t Dataword()            const { return dataword;             };      
      
    private:
      int me_bxn;
      int wire;
      int strip;
      int quality;
      int clct_pattern;
      int csc_ID;
      int epc; 
      int station;
      int neighbor;
      int sector;
      int subsector;
      int ring;
      int tbin_num;
      int bc0; 
      int bxe; 
      int lr;
      int afff;
      int cik; 
      int nit; 
      int afef;
      int se;
      int sm;
      int af;
      int vp;
      int stub_num;
      int format_errors;
      uint64_t dataword;
      
    }; // End of class ME
    
    // Define a vector of ME
    typedef std::vector<ME> MECollection;

  } // End of namespace emtf
} // End of namespace l1t

#endif /* define __l1t_emtf_ME_h__ */
