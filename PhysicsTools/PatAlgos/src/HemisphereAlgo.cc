#include "PhysicsTools/PatAlgos/interface/HemisphereAlgo.h"

using namespace std;

using std::vector;
using std::cout;
using std::endl;

HemisphereAlgo::HemisphereAlgo(vector<float> Px_vector, vector<float> Py_vector, vector<float> Pz_vector,
            vector<float> E_vector, int seed_method, int hemisphere_association_method) : Object_Px(Px_vector),
	    Object_Py(Py_vector), Object_Pz(Pz_vector), Object_E(E_vector), seed_meth(seed_method), 
	    hemi_meth(hemisphere_association_method), status(0), dbg(0) {
    for(int i = 0; i < (int) Object_Px.size(); i++){
      Object_Noseed.push_back(0);
    }
}   

HemisphereAlgo::HemisphereAlgo(vector<float> Px_vector, vector<float> Py_vector, vector<float> Pz_vector,
            vector<float> E_vector) : Object_Px(Px_vector),
	    Object_Py(Py_vector), Object_Pz(Pz_vector), Object_E(E_vector), seed_meth(0), 
	    hemi_meth(0), status(0), dbg(0) {
    for(int i = 0; i < (int) Object_Px.size(); i++){
      Object_Noseed.push_back(0);
    }
}   
   
vector<float> HemisphereAlgo::getAxis1(){
    if (status != 1){this->reconstruct();}   
    return Axis1;
}   
vector<float> HemisphereAlgo::getAxis2(){
    if (status != 1){this->reconstruct();}  
    return Axis2;
}

vector<int> HemisphereAlgo::getGrouping(){
    if (status != 1){this->reconstruct();}  
    return Object_Group;
}

int HemisphereAlgo::reconstruct(){

   
   int vsize = (int) Object_Px.size();
   if((int) Object_Py.size() != vsize || (int) Object_Pz.size() != vsize){
     cout << "WARNING!!!!! Input vectors have different size! Fix it!" << endl;
     return 0;
   }
   if (dbg > 0) {
//    cout << " HemisphereAlgo method, vsn = " << hemivsn << endl;
    cout << " HemisphereAlgo method " << endl;
   }
  
   if (!Object_P.empty()){
     Object_P.clear();
     Object_Pt.clear();
     Object_Eta.clear();
     Object_Phi.clear();
     Object_Group.clear();
     Axis1.clear();
     Axis2.clear();
   }
   for(int j = 0; j < vsize; j++){
     Object_P.push_back(0);
     Object_Pt.push_back(0);
     Object_Eta.push_back(0);
     Object_Phi.push_back(0);
     Object_Group.push_back(0);
   }
   for(int j = 0; j < 5; j++){
     Axis1.push_back(0);
     Axis2.push_back(0);
   }
  
   float theta;
   for (int i = 0; i <vsize; i++){
     Object_P[i] = sqrt(Object_Px[i]*Object_Px[i]+Object_Py[i]*Object_Py[i]+Object_Pz[i]*Object_Pz[i]);
     if (Object_P[i] > Object_E[i]+0.001) {  
      cout << "WARNING!!!!! Object " << i << " has E = " << Object_E[i]
      << " less than P = " << Object_P[i] << endl;
      return 0;
     } 
     Object_Pt[i] = sqrt(Object_Px[i]*Object_Px[i]+Object_Py[i]*Object_Py[i]);
     // protection for div by 0
     if (fabs(Object_Pz[i]) > 0.001) {
      theta = atan(sqrt(Object_Px[i]*Object_Px[i]+Object_Py[i]*Object_Py[i])/Object_Pz[i]);
     }
     else {
      theta = 1.570796327;
     }
     if (theta < 0.) {theta = theta + 3.141592654;}
     Object_Eta[i] = -log(tan(0.5*theta));
     Object_Phi[i] = atan2(Object_Py[i], Object_Px[i]);
     if (dbg > 0) {
      cout << " Object " << i << " Eta = " << Object_Eta[i]
                              << " Phi = " << Object_Phi[i] << endl;
     }
   }
   
   
   if (dbg > 0) {
    cout << endl;
    cout << " Seeding method = " << seed_meth << endl;
   }
   int I_Max = -1;
   int J_Max = -1;
   
   if (seed_meth == 1) {
    
    float P_Max = 0.;
    float DeltaRP_Max = 0.;
       
    // take highest momentum object as first axis   
    for (int i=0;i<vsize;i++){    
     Object_Group[i] = 0;
     //cout << "Object_Px[i] = " << Object_Px[i] << ", Object_Py[i] = " << Object_Py[i] 
     //<< ", Object_Pz[i] = " << Object_Pz[i] << "  << endl;    
     if (Object_Noseed[i] == 0 && P_Max < Object_P[i]){
       P_Max = Object_P[i];
       I_Max = i; 
     }           
    }
    
    Axis1[0] = Object_Px[I_Max] /  Object_P[I_Max];
    Axis1[1] = Object_Py[I_Max] /  Object_P[I_Max];
    Axis1[2] = Object_Pz[I_Max] /  Object_P[I_Max];
    Axis1[3] = Object_P[I_Max];
    Axis1[4] = Object_E[I_Max];
    
    // take as second axis
    for (int i=0;i<vsize;i++){           
      float DeltaR = sqrt((Object_Eta[i] - Object_Eta[I_Max])*(Object_Eta[i] - Object_Eta[I_Max]) 
      + (DeltaPhi(Object_Phi[i], Object_Phi[I_Max]))*(DeltaPhi(Object_Phi[i], Object_Phi[I_Max]))  );      
      if (Object_Noseed[i] == 0 && DeltaR > 0.5) {     
      float DeltaRP = DeltaR * Object_P[i];       
        if (DeltaRP > DeltaRP_Max){
	 DeltaRP_Max = DeltaRP;
	 J_Max = i;
	}	
      }      
    } 
    
    if (J_Max >=0){
     Axis2[0] = Object_Px[J_Max] /  Object_P[J_Max];
     Axis2[1] = Object_Py[J_Max] /  Object_P[J_Max];
     Axis2[2] = Object_Pz[J_Max] /  Object_P[J_Max];
     Axis2[3] = Object_P[J_Max];
     Axis2[4] = Object_E[J_Max];
     
    } else {   
      // cout << " This is a MONOJET." << endl;
    return 0;
    }
    if (dbg > 0) {
     cout << " Axis 1 is Object = " << I_Max << endl;
     cout << " Axis 2 is Object = " << J_Max << endl;
    }

   
   } else if (seed_meth == 2 | seed_meth == 3) {

    float Mass_Max = 0.;
    float InvariantMass = 0.;
    
    // maximize the invariant mass of two objects
    for (int i=0;i<vsize;i++){    
     Object_Group[i] = 0;
     if (Object_Noseed[i] == 0){ 
      for (int j=i+1;j<vsize;j++){  
       if (Object_Noseed[j] == 0){ 
         // either the invariant mass
         if (seed_meth == 2){
           InvariantMass =  (Object_E[i] + Object_E[j])* (Object_E[i] + Object_E[j])
	   - (Object_Px[i] + Object_Px[j])* (Object_Px[i] + Object_Px[j]) 
	   - (Object_Py[i] + Object_Py[j])* (Object_Py[i] + Object_Py[j])
	   - (Object_Pz[i] + Object_Pz[j])* (Object_Pz[i] + Object_Pz[j]) ;  
         } 
         // or the transverse mass
         else if (seed_meth == 3){
           float pti = sqrt(Object_Px[i]*Object_Px[i] + Object_Py[i]*Object_Py[i]);
           float ptj = sqrt(Object_Px[j]*Object_Px[j] + Object_Py[j]*Object_Py[j]);
           InvariantMass =  2. * (pti*ptj - Object_Px[i]*Object_Px[j]
                                          - Object_Py[i]*Object_Py[j] );
         }
         if ( Mass_Max < InvariantMass){
           Mass_Max = InvariantMass;
           I_Max = i;
           J_Max = j;
         }
        }               
       }
      }
    }
    
    if (J_Max>0) {

    Axis1[0] = Object_Px[I_Max] /  Object_P[I_Max];
    Axis1[1] = Object_Py[I_Max] /  Object_P[I_Max];
    Axis1[2] = Object_Pz[I_Max] /  Object_P[I_Max];
    
    Axis1[3] = Object_P[I_Max];
    Axis1[4] = Object_E[I_Max]; 
  
    Axis2[0] = Object_Px[J_Max] /  Object_P[J_Max];
    Axis2[1] = Object_Py[J_Max] /  Object_P[J_Max];
    Axis2[2] = Object_Pz[J_Max] /  Object_P[J_Max];
    
    Axis2[3] = Object_P[J_Max];
    Axis2[4] = Object_E[J_Max]; 

    } else {
      // cout << " This is a MONOJET." << endl;
      return 0;
    }
    if (dbg > 0) {
     cout << " Axis 1 is Object = " << I_Max << endl;
     cout << " Axis 2 is Object = " << J_Max << endl;
    }
    
    
   } else {
     cout << "Please give a valid seeding method!" << endl;
     return 0;
   }
   
   // seeding done 
   // now do the hemisphere association
   
    if (dbg > 0) {
     cout << endl;
     cout << " Association method = " << hemi_meth << endl;
    }

    int numLoop = 0;
    bool I_Move = true;


    while (I_Move && (numLoop < 100)){

    I_Move = false;
    numLoop++;
    if (dbg > 0) {
     cout << " Iteration = " << numLoop << endl;
    }
   
    float Sum1_Px = 0.;
    float Sum1_Py = 0.;
    float Sum1_Pz = 0.;
    float Sum1_P = 0.;
    float Sum1_E = 0.; 
    float Sum2_Px = 0.;
    float Sum2_Py = 0.;
    float Sum2_Pz = 0.;
    float Sum2_P = 0.;
    float Sum2_E = 0.; 
   
    
    if (hemi_meth == 1) {
    
      
   
    
     for (int i=0;i<vsize;i++){  
      float  P_Long1 = Object_Px[i]*Axis1[0] + Object_Py[i]*Axis1[1] + Object_Pz[i]*Axis1[2];
      float  P_Long2 = Object_Px[i]*Axis2[0]+ Object_Py[i]*Axis2[1] + Object_Pz[i]*Axis2[2];
      if (P_Long1 >= P_Long2){
          if (Object_Group[i] != 1){ 
	    I_Move = true;
	  }      
          Object_Group[i] = 1;
	  Sum1_Px += Object_Px[i];
	  Sum1_Py += Object_Py[i];
	  Sum1_Pz += Object_Pz[i];
	  Sum1_P += Object_P[i];
	  Sum1_E += Object_E[i]; 
      } else {
          if (Object_Group[i] != 2){ 
	    I_Move = true;
	  }
          Object_Group[i] = 2;
	  Sum2_Px += Object_Px[i];
	  Sum2_Py += Object_Py[i];
	  Sum2_Pz += Object_Pz[i];
	  Sum2_P += Object_P[i];
	  Sum2_E += Object_E[i]; 
      }
     }
    
    } else if (hemi_meth == 2 || hemi_meth == 3){
    
       for (int i=0;i<vsize;i++){  
        if (i == I_Max) {
	  Object_Group[i] = 1;
	  Sum1_Px += Object_Px[i];
	  Sum1_Py += Object_Py[i];
	  Sum1_Pz += Object_Pz[i];
	  Sum1_P += Object_P[i];
	  Sum1_E += Object_E[i]; 
	} else if (i == J_Max) {
	  Object_Group[i] = 2;
	  Sum2_Px += Object_Px[i];
	  Sum2_Py += Object_Py[i];
	  Sum2_Pz += Object_Pz[i];
	  Sum2_P += Object_P[i];
	  Sum2_E += Object_E[i]; 
        } else {
	
	
	 if(!I_Move){ 
	  
	 float NewAxis1_Px = Axis1[0] * Axis1[3];
	 float NewAxis1_Py = Axis1[1] * Axis1[3];
	 float NewAxis1_Pz = Axis1[2] * Axis1[3];
	 float NewAxis1_E = Axis1[4];
	 
         float NewAxis2_Px = Axis2[0] * Axis2[3];
         float NewAxis2_Py = Axis2[1] * Axis2[3];
         float NewAxis2_Pz = Axis2[2] * Axis2[3];
         float NewAxis2_E = Axis2[4];
       
         if (Object_Group[i] == 1){
	  
	  NewAxis1_Px = NewAxis1_Px - Object_Px[i];
	  NewAxis1_Py = NewAxis1_Py - Object_Py[i];
	  NewAxis1_Pz = NewAxis1_Pz - Object_Pz[i];
	  NewAxis1_E = NewAxis1_E - Object_E[i]; 
	 
	 } else if (Object_Group[i] == 2) {
	 
	  NewAxis2_Px = NewAxis2_Px - Object_Px[i];
	  NewAxis2_Py = NewAxis2_Py - Object_Py[i];
	  NewAxis2_Pz = NewAxis2_Pz - Object_Pz[i];
	  NewAxis2_E = NewAxis2_E - Object_E[i];
	 }
               
	  
         float mass1 =  NewAxis1_E - ((Object_Px[i]*NewAxis1_Px + Object_Py[i]*NewAxis1_Py +
	 Object_Pz[i]*NewAxis1_Pz)/Object_P[i]);
	 
	 float mass2 =  NewAxis2_E - ((Object_Px[i]*NewAxis2_Px + Object_Py[i]*NewAxis2_Py +
	 Object_Pz[i]*NewAxis2_Pz)/Object_P[i]);
	 
	 if (hemi_meth == 3) {
	 
	   mass1 *= NewAxis1_E/((NewAxis1_E+Object_E[i])*(NewAxis1_E+Object_E[i]));
	 
           mass2 *= NewAxis2_E/((NewAxis2_E+Object_E[i])*(NewAxis1_E+Object_E[i]));
	
	 }
	 
        if(mass1<mass2) {
	 if (Object_Group[i] != 1){ 
	    I_Move = true;
	  }
	  Object_Group[i] = 1;
       
          Sum1_Px += Object_Px[i];
	  Sum1_Py += Object_Py[i];
	  Sum1_Pz += Object_Pz[i];
	  Sum1_P += Object_P[i];
	  Sum1_E += Object_E[i]; 
         } else {
	  if (Object_Group[i] != 2){ 
	    I_Move = true;
	  }
	  Object_Group[i] = 2;
	  Sum2_Px += Object_Px[i];
	  Sum2_Py += Object_Py[i];
	  Sum2_Pz += Object_Pz[i];
	  Sum2_P += Object_P[i];
	  Sum2_E += Object_E[i]; 
	 
	 }
      
      
        } else {
	
	  if (Object_Group[i] == 1){
	     Sum1_Px += Object_Px[i];
	     Sum1_Py += Object_Py[i];
	     Sum1_Pz += Object_Pz[i];
	     Sum1_P += Object_P[i];
	     Sum1_E += Object_E[i]; 
	  }
	  if (Object_Group[i] == 2){
	     Sum2_Px += Object_Px[i];
	     Sum2_Py += Object_Py[i];
	     Sum2_Pz += Object_Pz[i];
	     Sum2_P += Object_P[i];
	     Sum2_E += Object_E[i]; 
	  }
         
	
	
	}
	
	
      }
     }
    } else {
      cout << "Please give a valid hemisphere association method!" << endl;
      return 0;
    }
    
    // recomputing the axes     

    Axis1[3] = sqrt(Sum1_Px*Sum1_Px + Sum1_Py*Sum1_Py + Sum1_Pz*Sum1_Pz);
    if (Axis1[3]<0.0001) {
      cout << "ZERO objects in group 1! " << endl; 
    } else {    
    Axis1[0] = Sum1_Px / Axis1[3];   
    Axis1[1] = Sum1_Py / Axis1[3];
    Axis1[2] = Sum1_Pz / Axis1[3];
    Axis1[4] = Sum1_E; 
    }
    
   
    
    Axis2[3] = sqrt(Sum2_Px*Sum2_Px + Sum2_Py*Sum2_Py + Sum2_Pz*Sum2_Pz);
    if (Axis2[3]<0.0001) {
      cout << " ZERO objects in group 2! " << endl; 
    } else {  
    Axis2[0] = Sum2_Px / Axis2[3];   
    Axis2[1] = Sum2_Py / Axis2[3];
    Axis2[2] = Sum2_Pz / Axis2[3];
    Axis2[4] = Sum2_E; 
    }

    if (dbg > 0) {
     cout << " Grouping = ";
     for (int i=0;i<vsize;i++){  
      cout << "  " << Object_Group[i];
     }
     cout << endl;
    }
    
    
    }
  
        
   status = 1;
   return 1;
}




float HemisphereAlgo::DeltaPhi(float v1, float v2)
 {
  float diff = fabs(v2 - v1);
  float corr = 2*acos(-1.) - diff;
  if (diff < acos(-1.)){ return diff;} else { return corr;} 
 }



