// Class for Event Record Header

#ifndef __l1t_emtf_EventHeader_h__
#define __l1t_emtf_EventHeader_h__

#include <boost/cstdint.hpp> 

namespace l1t {
  namespace emtf {
    class EventHeader {
    public:
      
      explicit EventHeader(uint64_t dataword);
    
    EventHeader() : 
      l1a(-99), l1a_BXN(-99), sp_TS(-99), endcap(-99), sector(-99), sp_ersv(-99), sp_addr(-99), 
	tbin(-99), ddm(-99), spa(-99), rpca(-99), skip(-99), rdy(-99), bsy(-99), osy(-99), 
	wof(-99), me1a(-99), me1b(-99), me2(-99), me3(-99), me4(-99), format_errors(0), dataword(-99) 
	{};
      
    EventHeader(int int_l1a, int int_l1a_BXN, int int_sp_TS, int int_endcap, int int_sector, int int_sp_ersv, int int_sp_addr, 
		int int_tbin, int int_ddm, int int_spa, int int_rpca, int int_skip, int int_rdy, int int_bsy, int int_osy, 
		int int_wof, int int_me1a, int int_me1b, int int_me2, int int_me3, int int_me4) :
      l1a(int_l1a), l1a_BXN(int_l1a_BXN), sp_TS(int_sp_TS), endcap(int_endcap), sector(int_sector), sp_ersv(int_sp_ersv), sp_addr(int_sp_addr), 
	tbin(int_tbin), ddm(int_ddm), spa(int_spa), rpca(int_rpca), skip(int_skip), rdy(int_rdy), bsy(int_bsy), osy(int_osy), 
	wof(int_wof), me1a(int_me1a), me1b(int_me1b), me2(int_me2), me3(int_me3), me4(int_me4), format_errors(0), dataword(-99)
    	{};
      
      virtual ~EventHeader() {};
      
      void set_l1a(int bits)           { l1a = bits;         }
      void set_l1a_BXN(int bits)       { l1a_BXN = bits;     }
      void set_sp_TS(int bits)         { sp_TS = bits;       }
      void set_endcap(int bits)        { endcap = bits;      }
      void set_sector(int bits)        { sector = bits;      }
      void set_sp_ersv(int bits)       { sp_ersv = bits;     }
      void set_sp_addr(int bits)       { sp_addr = bits;     }
      void set_tbin(int bits)          { tbin = bits;        }
      void set_ddm(int bits)           { ddm = bits;         }
      void set_spa(int bits)           { spa = bits;         }
      void set_rpca(int bits)          { rpca = bits;        }
      void set_skip(int bits)          { skip = bits;        }
      void set_rdy(int bits)           { rdy = bits;         }
      void set_bsy(int bits)           { bsy = bits;         }
      void set_osy(int bits)           { osy = bits;         }
      void set_wof(int bits)           { wof = bits;         }
      void set_me1a(int bits)          { me1a = bits;        }
      void set_me1b(int bits)          { me1b = bits;        }
      void set_me2(int bits)           { me2 = bits;         }
      void set_me3(int bits)           { me3 = bits;         }
      void set_me4(int bits)           { me4 = bits;         }
      void add_format_error()          { format_errors += 1; }
      void set_dataword(uint64_t bits) { dataword = bits;    }
      
      int      L1A()           const { return l1a;           }
      int      L1A_BXN()       const { return l1a_BXN;       }
      int      SP_TS()         const { return sp_TS;         }
      int      Endcap()        const { return endcap;        }
      int      Sector()        const { return sector;        }
      int      SP_ersv()       const { return sp_ersv;       }
      int      SP_addr()       const { return sp_addr;       }
      int      TBIN()          const { return tbin;          }
      int      DDM()           const { return ddm;           }
      int      SPa()           const { return spa;           }
      int      RPCa()          const { return rpca;          }
      int      Skip()          const { return skip;          }
      int      Rdy()           const { return rdy;           }
      int      BSY()           const { return bsy;           }
      int      OSY()           const { return osy;           }
      int      WOF()           const { return wof;           }
      int      ME1a()          const { return me1a;          }
      int      ME1b()          const { return me1b;          }
      int      ME2()           const { return me2;           }
      int      ME3()           const { return me3;           }
      int      ME4()           const { return me4;           }
      int      Format_Errors() const { return format_errors; }
      uint64_t Dataword()      const { return dataword;      }      
      
    private:
      int l1a;
      int l1a_BXN;
      int sp_TS;
      int endcap;
      int sector;
      int sp_ersv;
      int sp_addr;
      int tbin;
      int ddm; 
      int spa; 
      int rpca;
      int skip;
      int rdy; 
      int bsy; 
      int osy; 
      int wof; 
      int me1a;
      int me1b;
      int me2; 
      int me3; 
      int me4; 
      int format_errors;
      uint64_t dataword;
      
    }; // End of class EventHeader
  } // End of namespace emtf
} // End of namespace l1t

#endif /* define __l1t_emtf_EventHeader_h__ */
