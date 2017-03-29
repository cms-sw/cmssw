// Class for Sector Processor (SP) Output Data Record

#ifndef __l1t_emtf_SP_h__
#define __l1t_emtf_SP_h__

#include <boost/cstdint.hpp> 

namespace l1t {
  namespace emtf {
    class SP {
    public:
      
      explicit SP(uint64_t dataword);
    
    SP() :
      hl(-99), c(-99), phi_full(-99), vc(-99), vt(-99), se(-99), bc0(-99), 
	quality_GMT(-99), phi_GMT(-99), bx(-99), mode(-99), eta_GMT(-99), pt_GMT(-99),
	me1_subsector(-99), me1_CSC_ID(-99), me1_stub_num(-99), me2_CSC_ID(-99), 
	me2_stub_num(-99), me3_CSC_ID(-99), me3_stub_num(-99), me4_CSC_ID(-99), 
	me4_stub_num(-99), tbin(-99), me1_delay(-99), me2_delay(-99), me3_delay(-99), 
	me4_delay(-99), pt_LUT_addr(-99), format_errors(0), dataword(-99) 
	{};

      virtual ~SP() {};

      void set_hl (int bits)            { hl = bits;             }
      void set_c (int bits)             { c = bits;              }
      void set_phi_full (int bits)      { phi_full = bits;       }
      void set_vc (int bits)            { vc = bits;             }
      void set_vt (int bits)            { vt = bits;             }
      void set_se (int bits)            { se = bits;             }
      void set_bc0 (int bits)           { bc0 = bits;            }
      void set_quality_GMT (int bits)   { quality_GMT = bits;    }
      void set_phi_GMT (int bits)       { phi_GMT = bits;        }
      void set_bx (int bits)            { bx = bits;             }
      void set_mode (int bits)          { mode = bits;           }
      void set_eta_GMT (int bits)       { eta_GMT = bits;        }
      void set_pt_GMT (int bits)        { pt_GMT = bits;         }
      void set_me1_subsector (int bits) { me1_subsector = bits;  }
      void set_me1_CSC_ID (int bits)    { me1_CSC_ID = bits;     }
      void set_me1_stub_num (int bits)  { me1_stub_num = bits;   }      
      void set_me2_CSC_ID (int bits)    { me2_CSC_ID = bits;     }
      void set_me2_stub_num (int bits)  { me2_stub_num = bits;   }
      void set_me3_CSC_ID (int bits)    { me3_CSC_ID = bits;     }
      void set_me3_stub_num (int bits)  { me3_stub_num = bits;   }
      void set_me4_CSC_ID (int bits)    { me4_CSC_ID = bits;     }
      void set_me4_stub_num (int bits)  { me4_stub_num = bits;   }
      void set_tbin (int bits)          { tbin = bits;           }
      void set_me1_delay (int bits)     { me1_delay = bits;      }
      void set_me2_delay (int bits)     { me2_delay = bits;      }
      void set_me3_delay (int bits)     { me3_delay = bits;      }
      void set_me4_delay (int bits)     { me4_delay = bits;      }
      void set_pt_LUT_addr(unsigned long bits) { pt_LUT_addr = bits;   }
      void add_format_error()           { format_errors += 1;    }
      void set_dataword(uint64_t bits)  { dataword = bits;       }

      int HL()             const { return hl;             }
      int C()              const { return c;              }
      int Phi_full()       const { return phi_full;       }
      int VC()             const { return vc;             }
      int VT()             const { return vt;             }
      int SE()             const { return se;             }
      int BC0()            const { return bc0;            }
      int Quality_GMT()    const { return quality_GMT;    }
      int Phi_GMT()        const { return phi_GMT;        }
      int BX()             const { return bx;             }
      int Mode()           const { return mode;           }
      int Eta_GMT()        const { return eta_GMT;        }
      int Pt_GMT()         const { return pt_GMT;         }
      int ME1_subsector()  const { return me1_subsector;  }
      int ME1_CSC_ID()     const { return me1_CSC_ID;     }
      int ME1_stub_num()   const { return me1_stub_num;   }      
      int ME2_CSC_ID()     const { return me2_CSC_ID;     }
      int ME2_stub_num()   const { return me2_stub_num;   }      
      int ME3_CSC_ID()     const { return me3_CSC_ID;     }
      int ME3_stub_num()   const { return me3_stub_num;   }      
      int ME4_CSC_ID()     const { return me4_CSC_ID;     }
      int ME4_stub_num()   const { return me4_stub_num;   }      
      int TBIN()           const { return tbin;           }
      int ME1_delay()      const { return me1_delay;      }
      int ME2_delay()      const { return me2_delay;      }
      int ME3_delay()      const { return me3_delay;      }
      int ME4_delay()      const { return me4_delay;      }
      unsigned long Pt_LUT_addr() const { return pt_LUT_addr; }
      int Format_Errors()  const { return format_errors;  }
      uint64_t Dataword()  const { return dataword;       } 
      
    private:

      int hl;
      int c;
      int phi_full;
      int vc;
      int vt;
      int se;
      int bc0;
      int quality_GMT;
      int phi_GMT;
      int bx;
      int mode;
      int eta_GMT;
      int pt_GMT;
      int me1_subsector;
      int me1_CSC_ID;      
      int me1_stub_num;      
      int me2_CSC_ID;
      int me2_stub_num;
      int me3_CSC_ID;
      int me3_stub_num;
      int me4_CSC_ID;      
      int me4_stub_num;
      int tbin;
      int me1_delay;
      int me2_delay;
      int me3_delay;
      int me4_delay;
      unsigned long pt_LUT_addr;
      int format_errors;
      uint64_t dataword;
      
    }; // End of class SP

    // Define a vector of SP
    typedef std::vector<SP> SPCollection;

  } // End of namespace emtf
} // End of namespace l1t

#endif /* define __l1t_emtf_SP_h__ */
