void oriCode(int & binoffy, int & pitchMul) {
  int  m_pitchy=1;
  int local_pitchy=1;
  if (binoffy>416) {            // ROC 8, not real ROC
    binoffy=binoffy+17;
  } else if (binoffy==416) {    // ROC 8
    binoffy=binoffy+16;
    local_pitchy = 2 * m_pitchy;
    
  } else if (binoffy==415) {    // ROC 7, last big pixel
      binoffy=binoffy+15;
      local_pitchy = 2 * m_pitchy;
  } else if (binoffy>364) {     // ROC 7
    binoffy=binoffy+15;
  } else if (binoffy==364) {    // ROC 7
    binoffy=binoffy+14;
    local_pitchy = 2 * m_pitchy;
    
  } else if (binoffy==363) {      // ROC 6
    binoffy=binoffy+13;
    local_pitchy = 2 * m_pitchy;    
  } else if (binoffy>312) {       // ROC 6
    binoffy=binoffy+13;
  } else if (binoffy==312) {      // ROC 6
    binoffy=binoffy+12;
    local_pitchy = 2 * m_pitchy;
    
  } else if (binoffy==311) {      // ROC 5
    binoffy=binoffy+11;
    local_pitchy = 2 * m_pitchy;    
  } else if (binoffy>260) {       // ROC 5
    binoffy=binoffy+11;
  } else if (binoffy==260) {      // ROC 5
    binoffy=binoffy+10;
    local_pitchy = 2 * m_pitchy;
    
  } else if (binoffy==259) {      // ROC 4
    binoffy=binoffy+9;
    local_pitchy = 2 * m_pitchy;    
  } else if (binoffy>208) {       // ROC 4
      binoffy=binoffy+9;
  } else if (binoffy==208) {      // ROC 4
    binoffy=binoffy+8;
    local_pitchy = 2 * m_pitchy;
    
  } else if (binoffy==207) {      // ROC 3
    binoffy=binoffy+7;
      local_pitchy = 2 * m_pitchy;    
  } else if (binoffy>156) {       // ROC 3
    binoffy=binoffy+7;
  } else if (binoffy==156) {      // ROC 3
      binoffy=binoffy+6;
      local_pitchy = 2 * m_pitchy;
      
  } else if (binoffy==155) {      // ROC 2
    binoffy=binoffy+5;
      local_pitchy = 2 * m_pitchy;    
  } else if (binoffy>104) {       // ROC 2
    binoffy=binoffy+5;
  } else if (binoffy==104) {      // ROC 2
      binoffy=binoffy+4;
      local_pitchy = 2 * m_pitchy;
      
    } else if (binoffy==103) {      // ROC 1
    binoffy=binoffy+3;
    local_pitchy = 2 * m_pitchy;    
  } else if (binoffy>52) {       // ROC 1
      binoffy=binoffy+3;
  } else if (binoffy==52) {      // ROC 1
    binoffy=binoffy+2;
      local_pitchy = 2 * m_pitchy;
      
  } else if (binoffy==51) {      // ROC 0
      binoffy=binoffy+1;
      local_pitchy = 2 * m_pitchy;    
  } else if (binoffy>0) {        // ROC 0
    binoffy=binoffy+1;
    } else if (binoffy==0) {       // ROC 0
    binoffy=binoffy+0;
    local_pitchy = 2 * m_pitchy;
  }
  pitchMul=local_pitchy;
}


#include<algorithm>
void newCode(int & binoffy, int & pitchMul) {
  int offIndex[] = {0,51,52,103,104,155,156,207,208,259,260,311,312,363,364,415,416,511};
  pitchMul=1;
  auto const j = std::lower_bound(std::begin(offIndex),std::end(offIndex),binoffy);
  if (*j==binoffy) pitchMul=2;
  binoffy+= (j-offIndex);
}

#include<cstdio>
int main() {
  for (int i=0; i!=511; ++i) {
    int oldb=i; int newb=i; int op=0; int np=0;
    oriCode(oldb,op);
    newCode(newb,np);
    if (oldb!=newb || op!=np)
      printf("%d: %d,%d  %d,%d\n",i, oldb,newb,op,np);
  }
  return 0;
}
