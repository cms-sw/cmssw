#include <vector>
class CentralityTable {
  
  public:

    struct Bins {

         double hf_low_cut;
         double n_part_mean;
         double n_part_var;
         double n_coll_mean;
         double n_coll_var;  
         double b_mean;
         double b_var;   
     };
  
    CentralityTable(){}
  // CentralityTable(double hf, double npart, double npvar, double ncoll, double ncvar, double b, double bvar);
    virtual ~CentralityTable(){}
   
    std::vector<Bins> m_table;
   
};

