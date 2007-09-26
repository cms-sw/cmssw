#include "DefectsFromConstructionDB.h"
         


v_detidallbadstrips DefectsFromConstructionDB::GetBadStripsFromConstructionDB(){

  
  // variables for function
  p_channelflag p_chanflag;
  v_channelflag v_chanflag;
  
  unsigned int detid;
  short flag;
  short channel;
  



  // open filename from constructor
  ifstream infile(inputfile);
  if(!infile){std::cout << "Problem while trying to open File: " << inputfile << std::endl;}



  unsigned int detid_temp=0;
  bool first_detid = true;
  v_allbadstrips.clear();

  while(!infile.eof()){

    // get data from file: 
    infile >> detid >> channel >> flag;
    if(detid_temp==detid || first_detid){
      p_chanflag=p_channelflag(channel, flag);
      v_chanflag.push_back(p_chanflag);
    }
    else{
     
      v_allbadstrips.push_back(p_detidchannelflag(detid_temp,v_chanflag));
      v_chanflag.clear();
      p_chanflag=p_channelflag(channel, flag);
      v_chanflag.push_back(p_chanflag);
         }
   
    if(first_detid) first_detid = false;
    detid_temp=detid;

  }

 
  
  return v_allbadstrips;
 

}

void DefectsFromConstructionDB::print(){
   for(v_detidallbadstrips::iterator iter=v_allbadstrips.begin(); iter<v_allbadstrips.end();iter++){
   
    // print detid
    std::cout << (*iter).first<<"\t";

    // print corresponding badstrips with flags

    for(v_channelflag::iterator it=iter->second.begin(); it<iter->second.end(); it++){
     
      std::cout << (*it).first <<"\t" << (*it).second << "\t";

   }

    std::cout<< std::endl << std::endl;
   
  }


}



