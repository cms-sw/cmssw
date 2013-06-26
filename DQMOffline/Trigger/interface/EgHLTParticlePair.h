#ifndef DQMOFFLINE_TRIGGER_EGHLTPARTICLEPAIR
#define DQMOFFLINE_TRIGGER_EGHLTPARTICLEPAIR

//a useful struct for defining a pair of particles


namespace egHLT {
  template<class T> struct ParticlePair{
    const T& part1; 
    const T& part2;
    
    ParticlePair(const T& particle1,const T& particle2):part1(particle1),part2(particle2){}
    ~ParticlePair(){}
    
    float mass()const{return (part1.p4()+part2.p4()).mag();}
    
  };
}
#endif
