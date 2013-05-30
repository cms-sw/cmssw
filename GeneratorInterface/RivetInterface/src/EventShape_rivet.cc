#include "EventShape_rivet.hh"

using namespace std;

EventShape_rivet::EventShape_rivet(vector<HepLorentzVector> four_vector) : Object_4v(four_vector){}

vector<double> EventShape_rivet::getEventShapes(){
  this->calculate();
  return EventShapes;
}

vector<double> EventShape_rivet::getThrustAxis(){
  this->calculate();
  return ThrustAxis;
}

//vector<double> EventShape_rivet::getThrustAxis_C(){
//  this->calculate();
//  return ThrustAxis_c;
//}

int EventShape_rivet::calculate(){
  
  if (!Object_P.empty()){
    Object_P.clear();
    Object_Pt.clear();
    Object_Eta.clear();
    Object_Phi.clear();
    EventShapes.clear();
    ThrustAxis.clear();
    //    ThrustAxis_c.clear();
  }
  for(unsigned int j = 0; j < Object_4v.size(); j++){
    Object_P.push_back(0.);
    Object_Pt.push_back(0.);
    Object_Eta.push_back(0.);
    Object_Phi.push_back(0.);
  }

  for(int j = 0; j < nevtvar; j++){
    EventShapes.push_back(-50.);
  }

  EventShapes.push_back(double(Object_4v.size()));

  for(unsigned int j = 0; j < 3; j++){
    ThrustAxis.push_back(0.);
    //    ThrustAxis_c.push_back(0.);
  }
  
  double theta=0;

  for(unsigned int k =0; k<Object_4v.size(); k++){
    Object_P[k] = Object_4v[k].rho();
    Object_Pt[k] = Object_4v[k].perp();
    if(Object_4v[k].rho()>Object_4v[k].e() + 1E-4){
      cout << "ERROR!!! Object " << k <<" has P = " << Object_4v[k].rho() << " which is bigger than E = " << Object_4v[k].e() <<" "<<Object_4v[k].px()<<" "<<Object_4v[k].py()<<" "<<Object_4v[k].pz()<<" of total length "<< Object_4v.size()<<endl;
      return 0;}
    
    if (fabs(Object_4v[k].pz()) > 1E-5) {
      theta = atan(Object_4v[k].perp()/(Object_4v[k].pz()));
    } else {theta = M_PI/2;}
    
    if(theta<0.){theta = theta + M_PI;}
    Object_Eta[k] = - log(tan(0.5*theta));
    
    Object_Phi[k] = Object_4v[k].phi();
  }
  
  //sum of all Pt's
  double Pt_sum =0;
  for(unsigned int i=0;i<Object_4v.size();i++){
    Pt_sum+=Object_4v[i].perp();
  }

  //calculate the directly global transverse thrust dirgltrthr
  double dirgltrthr=0;

  std::vector<double> Thrust;
  Thrust = Thrust_calculate(Object_4v);
  //the variable thrust is tau = 1-thrust 
  dirgltrthr =Thrust[3];

  EventShapes[0] = dirgltrthr;  

  for(unsigned int k=0;k<3;k++){
    ThrustAxis[k] = Thrust[k];
  }
  
  //the directly global thrust minor dirglthrmin
  double dirglthrmin =0;
  //rotate the coordinate system around the beam axis that
  //the thrust axis is the new y'-Axis - the projections are
  //simply the new y-values then
  double alpha=atan2(ThrustAxis[1],ThrustAxis[0]);
  for(unsigned int i=0; i<Object_4v.size(); i++){
    dirglthrmin+= fabs(-sin(alpha)*Object_4v[i].px()+cos(alpha)*Object_4v[i].py());
  }
  dirglthrmin=dirglthrmin/Pt_sum;
  EventShapes[1] = dirglthrmin;


  double cenbroad_up=0;
  double cenbroad_down=0;
  
  double eta_up=0;
  unsigned int num_up=0;
  double eta_down =0;
  unsigned int num_down =0;
  double phi_temp =0;
  double phi_up_aver =0;
  double phi_down_aver =0;
  double Pt_sum_up =0;
  double Pt_sum_down =0;
  double dot_product_b =0;
  vector<double> phi_up;
  vector<double> phi_down;
  double py_rot =0;
  double px_rot =0;


  for(unsigned int i=0;i<Object_4v.size();i++){
    //    Pt_sum+=Object_4v[i].perp();
    dot_product_b =(Object_4v[i].px()*ThrustAxis[0]+Object_4v[i].py()*ThrustAxis[1]);
      if(dot_product_b>=0){
	Pt_sum_up+=Object_4v[i].perp();
	//rotate the coordinate system so that
	//the central thrust axis is e_x
        px_rot = cos(alpha)*Object_4v[i].px()+sin(alpha)*Object_4v[i].py();
        py_rot = - sin(alpha)*Object_4v[i].px()+cos(alpha)*Object_4v[i].py();
        //calculate the eta and phi in the rotated system
        eta_up+=Object_4v[i].perp()*Object_4v[i].eta();
        phi_temp =atan2(py_rot,px_rot);
	
	if(phi_temp>M_PI/2){
	  phi_temp = phi_temp - M_PI/2;}
	if (phi_temp<-M_PI/2){
	  phi_temp = phi_temp + M_PI/2;
	  }
	phi_up.push_back(phi_temp);
	phi_up_aver+=Object_4v[i].perp()*phi_temp;
	num_up+=1;
      }else{
	eta_down+=Object_4v[i].perp()*Object_4v[i].eta();
	Pt_sum_down+=Object_4v[i].perp();
	px_rot = cos(alpha)*Object_4v[i].px()+sin(alpha)*Object_4v[i].py();
	py_rot = - sin(alpha)*Object_4v[i].px()+cos(alpha)*Object_4v[i].py();
	phi_temp =atan2(py_rot,px_rot);
	if(phi_temp>M_PI/2){
          //if phi is bigger than pi/2 in the new system calculate 
          //the difference to the thrust axis 
	  phi_temp = M_PI -phi_temp;}
	if (phi_temp<-M_PI/2){
          //if phi is smaller than 
	  phi_temp = -M_PI-phi_temp;}
	phi_down.push_back(phi_temp);
        //calculate the pt-weighted phi
	phi_down_aver+=Object_4v[i].perp()*phi_temp;	
	num_down+=1;
      }
  }
  if (num_up!=0){
    eta_up=eta_up/Pt_sum_up;
    phi_up_aver=phi_up_aver/Pt_sum_up;}
  if(num_down!=0){
    eta_down = eta_down/Pt_sum_down;
    phi_down_aver=phi_down_aver/Pt_sum_down;}
  
  unsigned int index_up=0;
  unsigned int index_down=0;
  
  for(unsigned int i=0;i<Object_4v.size();i++){
    dot_product_b =Object_4v[i].px()*ThrustAxis[0]+Object_4v[i].py()*ThrustAxis[1];
    if(dot_product_b>=0){
      //calculate the broadenings of the regions with the rotated system
      //and the pt-weighted average of phi in the rotated system
      cenbroad_up+=Object_4v[i].perp()*sqrt(pow(Object_4v[i].eta()-eta_up,2)+pow(DeltaPhi(phi_up[index_up],phi_up_aver),2));
      index_up+=1;
    }else{
      cenbroad_down+=Object_4v[i].perp()*sqrt(pow(Object_4v[i].eta()-eta_down,2)+pow(DeltaPhi(phi_down[index_down],phi_down_aver),2));
      index_down+=1;
    }
  }
  if (index_up == 0 || index_down ==0) EventShapes[nevtvar] *=-1.;
  
  cenbroad_up=cenbroad_up/(2*Pt_sum);
  cenbroad_down=cenbroad_down/(2*Pt_sum);
  
  //central total jet broadening
  double centotbroad=0;
  centotbroad = cenbroad_up+cenbroad_down;
  
  EventShapes[2]=centotbroad;
  //    double cenwidbroad=0;
  //    cenwidbroad = max(cenbroad_up,cenbroad_down);
  
  //    EventShapes[3]=cenwidbroad;
  
  for (int ij=0; ij<nevtvar; ij++) {
    if (EventShapes[ij] < 1.e-20) EventShapes[ij] = 1.e-20;
    EventShapes[ij] = log(EventShapes[ij]);
  }

  return 1;
}


std::vector<double> EventShape_rivet::Thrust_calculate (std::vector<HepLorentzVector> Input_4v) {

  std::vector<double> Input_Px;
  std::vector<double> Input_Py;
  double thrustmax_calc =0;
  double temp_calc =0;
  unsigned int length_thrust_calc =0;
  std::vector<double> ThrustValues;
  std::vector<double> Thrust_Axis_calc;
  std::vector<double> p_thrust_max_calc;
  std::vector<double> p_dec_1_calc;
  std::vector<double> p_dec_2_calc;
  std::vector<double> p_pt_beam_calc;

   if (!ThrustValues.empty()){
     ThrustValues.clear();
     Thrust_Axis_calc.clear();
     p_thrust_max_calc.clear();
     p_dec_1_calc.clear();
     p_dec_2_calc.clear();
     p_pt_beam_calc.clear();
   }

  for(unsigned int j = 0; j < 3; j++){
    p_pt_beam_calc.push_back(0.);
    p_dec_1_calc.push_back(0.);
    p_dec_2_calc.push_back(0.);
    p_thrust_max_calc.push_back(0.);
    Thrust_Axis_calc.push_back(0.);
  }
  
  for(unsigned int j =0;j<4;j++){
    ThrustValues.push_back(0.);
  }

  //  cout <<"lenght "<< Input_Px.size()<<endl;
  length_thrust_calc = Input_4v.size();
  for (unsigned k=0; k<length_thrust_calc; k++) {
    Input_Px.push_back(Input_4v[k].px());
    Input_Py.push_back(Input_4v[k].py());
  }

  float Pt_sum_calc =0;

  for(unsigned int k=0;k<length_thrust_calc;k++){
    Pt_sum_calc+=sqrt(pow(Input_Px[k],2)+pow(Input_Py[k],2)); 
    for(unsigned int j = 0; j < 3; j++){
      p_thrust_max_calc[j]=0;
    }
    //get a vector perpendicular to the beam axis and 
    //perpendicular to the momentum of particle k
    //per default beam axis b = (0,0,1)   
    p_pt_beam_calc[0] = Input_Py[k]*1; 
    p_pt_beam_calc[1] = - Input_Px[k]*1;
    p_pt_beam_calc[2] = 0.; // GMA p_pt_beam_calc[3] = 0.;
    for(unsigned int i=0;i<length_thrust_calc;i++){
      if(i!=k){
	if((Input_Px[i]*p_pt_beam_calc[0]+Input_Py[i]*p_pt_beam_calc[1])>=0){
	  p_thrust_max_calc[0]= p_thrust_max_calc[0]+Input_Px[i];
	  p_thrust_max_calc[1]= p_thrust_max_calc[1]+Input_Py[i];
	}
	else{
	  p_thrust_max_calc[0]= p_thrust_max_calc[0]-Input_Px[i];
	  p_thrust_max_calc[1]= p_thrust_max_calc[1]-Input_Py[i];
	}
      }
    }
    p_dec_1_calc[0]=p_thrust_max_calc[0]+Input_Px[k];
    p_dec_1_calc[1]=p_thrust_max_calc[1]+Input_Py[k];
    p_dec_1_calc[2]=0;
    p_dec_2_calc[0]=p_thrust_max_calc[0]-Input_Px[k];
    p_dec_2_calc[1]=p_thrust_max_calc[1]-Input_Py[k];
    p_dec_2_calc[2]=0;
    temp_calc = pow(p_dec_1_calc[0],2)+pow(p_dec_1_calc[1],2);

    if(temp_calc>thrustmax_calc){
      thrustmax_calc =temp_calc;
      for(unsigned int i=0;i<3;i++){
	Thrust_Axis_calc[i]=p_dec_1_calc[i]/sqrt(thrustmax_calc);
      }
    }
    temp_calc = pow(p_dec_2_calc[0],2)+pow(p_dec_2_calc[1],2);
    if(temp_calc>thrustmax_calc){
      thrustmax_calc =temp_calc;
      for(unsigned int i=0;i<3;i++){
	Thrust_Axis_calc[i]=p_dec_2_calc[i]/sqrt(thrustmax_calc);
      }
    }
  }
  for(unsigned int j=0;j<3;j++){
    ThrustValues[j]=Thrust_Axis_calc[j];}
  double thrust_calc=0;
  thrust_calc = sqrt(thrustmax_calc)/Pt_sum_calc;
  
  //the variable which gets resummed is not the thrust
  //but tau=1-thrust
  ThrustValues[3]=1.-thrust_calc; 
  
  if (ThrustValues[3] < 1.e-20) ThrustValues[3] = 1.e-20;

  return ThrustValues;
}


double EventShape_rivet::DeltaPhi(double phi1, double phi2)
{
  double delta_phi = fabs(phi2 - phi1);
  if (delta_phi > M_PI){ 
    delta_phi = 2*M_PI-delta_phi;
  } 
  return delta_phi;
} 
