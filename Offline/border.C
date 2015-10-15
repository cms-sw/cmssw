#include <fstream>
#include <algorithm>
#include <iostream>

bool sortByX(const std::pair<int, int>& lx, const std::pair<int, int>& rx){ return lx.first < rx.first; }
bool sortByY(const std::pair<int, int>& lx, const std::pair<int, int>& rx){ return lx.second < rx.second; }

void border(){

    std::ifstream StripPos;
    StripPos.open("./stripID.dat");
    std::ofstream fvline("./borderv.dat", std::ofstream::out);
    std::ofstream fhline("./borderh.dat", std::ofstream::out);
    int ix[] = {0,0,0,0,0}, iy[] = {0,0,0,0,0};
    int iDCC(0), iTT(0), iST(0);
    std::vector<pair<int, int>> crystalID;
    
    for(unsigned line(0); line < 50401; line++){
      StripPos >>iDCC >> iTT >> iST >>  ix [0] >> iy[0] >> ix[1] >> iy[1] >> ix[2] >> iy[2] >> ix[3] >> iy[3] >> ix[4] >> iy[4];
      if(ix[0]==0 && ix[1]==0 && ix[2] ==0 && ix[3]==0)continue;
      crystalID.clear();
      for(unsigned i(0); i<5; i++)crystalID.push_back(make_pair(ix[i],iy[i]));
    
      //std::cout << iDCC+1 << " " << iTT+1 << " " << iST+1; 
      sort(crystalID.begin(), crystalID.end(),sortByY);
      //for(unsigned i(0); i<5; i++)std::cout << " " << crystalID[i].first << "," << crystalID[i].second << " ";
      //std::cout << std::endl;
     
      int tmpY(0),minX(0);
      tmpY = crystalID[0].second;
      minX = crystalID[0].first;
      std::vector<pair<int, int>> tmpLine;
      tmpLine.clear();
      for(unsigned aa(0); aa< 5; aa++){
        if(crystalID[aa].second == tmpY){
           minX = minX < crystalID[aa].first? minX: crystalID[aa].first;
        }
        else {
           tmpLine.push_back(make_pair(minX, tmpY));  
           tmpY = crystalID[aa].second;
           minX = crystalID[aa].first;
        }
        if(aa==4)tmpLine.push_back(make_pair(minX, tmpY));
      }
       
     // for(unsigned bb(0); bb<tmpLine.size(); bb++){
     //     std::cout << tmpLine[bb].first << "," << tmpLine[bb].second << " ";
     //}
     //std::cout << std::endl;
     //std::cout << "*********************************************" << std::endl;

      std::vector<int> position; 
      position.push_back(tmpLine[0].first); 
      position.push_back(tmpLine[0].second-1);
      position.push_back(tmpLine[0].second);
  
      for(unsigned iL(0); iL < tmpLine.size(); iL++){
         if(tmpLine[iL].first == position[0]){ 
           position[2] = tmpLine[iL].second;
         }
         else{
           fvline << position[0]-1 << "," << position[1] << "," << position[2] << std::endl; 
           position[0] = tmpLine[iL].first;
           position[1] = tmpLine[iL].second-1;
           position[2] = tmpLine[iL].second;
         }
         if(iL == tmpLine.size()-1)fvline << position[0]-1 << "," << position[1] << "," << position[2] << std::endl;
      }
   
      //std::cout << std::endl;
      //std::cout << "*********************************************" << std::endl;
      sort(crystalID.begin(), crystalID.end(),sortByX);
     
      int tmpX(0),minY(0);
      tmpX = crystalID[0].first;
      minY = crystalID[0].second;
      tmpLine.clear();
      for(unsigned aa(0); aa< 5; aa++){
        if(crystalID[aa].first == tmpX){
           minY = minY < crystalID[aa].second? minY: crystalID[aa].second;
        }
        else {
           tmpLine.push_back(make_pair(tmpX, minY));  
           tmpX = crystalID[aa].first;
           minY = crystalID[aa].second;
        }
        if(aa==4)tmpLine.push_back(make_pair(tmpX,minY));
      }
       
      position.clear(); 
      position.push_back(tmpLine[0].first-1); 
      position.push_back(tmpLine[0].first);
      position.push_back(tmpLine[0].second);
  
      for(unsigned iL(0); iL < tmpLine.size(); iL++){
         if(tmpLine[iL].second == position[2]){ 
           position[1] = tmpLine[iL].first;
         }
         else{
           fhline << position[2]-1 << "," << position[0] << "," << position[1] << std::endl; 
           position[0] = tmpLine[iL].first-1;
           position[1] = tmpLine[iL].first;
           position[2] = tmpLine[iL].second;
         }
         if(iL == tmpLine.size()-1)fhline << position[2]-1 << "," << position[0] << "," << position[1] << std::endl;
      }
      }  
         

 }     
