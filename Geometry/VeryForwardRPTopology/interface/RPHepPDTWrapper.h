#ifndef Geometry_VeryForwardRPTopology_RPHepPDTWrapper
#define Geometry_VeryForwardRPTopology_RPHepPDTWrapper

#include <fstream>
#include <string>
#include "HepPDT/defs.h"
#include "HepPDT/TableBuilder.hh"
#include "HepPDT/ParticleDataTable.hh"
#include "HepMC/GenEvent.h"

#include "TH1F.h"

class RPHepPDTWrapper
{
  public:
    static inline double GetMass(int particle_id)
    {
      if(!datacol)
        init();

      const HepPDT::ParticleData *pd = datacol->particle( abs(particle_id) );
      if(pd)
        return pd->mass();
      else
        return -1;
    }
    
    static std::string GetName(int particle_id)
    {
      if(!datacol)
        init();

      const HepPDT::ParticleData *pd = datacol->particle( abs(particle_id) );
      if(pd)
      {
        if(particle_id<0)
      	{
      	  return pd->name() + std::string("_bar");
      	}
      	else
      	{
      	  return pd->name();
      	}
      }
      else
        return std::string();
    }
    
    static double GetCharge(int particle_id)
    {
      if(!datacol)
        init();

      const HepPDT::ParticleData *pd = datacol->particle( abs(particle_id) );
      if(pd)
      {
        if(particle_id<0)
        {
          return -pd->charge();
        }
        else
        {
          return pd->charge();
        }
      }
      else
        return 0;
    }
    
    static void SetBinLabels(TH1 & hist);
  private:
    static HepPDT::ParticleDataTable *datacol;
    
    RPHepPDTWrapper() {};
    static void init()
    {
        char *cmsswPath = getenv("CMSSW_RELEASE_BASE");
        std::string pdgfile = std::string(cmsswPath) + std::string("/src/SimGeneral/HepPDTESSource/data/PDG_mass_width_2004.mc");
      std::ifstream pdfile( pdgfile );
      if( !pdfile )
      { 
        std::cerr << "cannot open " << pdgfile << std::endl;
        exit(-1);
      }
      // construct empty PDT
      datacol = new HepPDT::ParticleDataTable("PDG Table");
      {
        HepPDT::TableBuilder  tb(*datacol);
        // read the input - put as many here as you want
        if(!addPDGParticles(pdfile, tb))
        {
          std::cout << "error reading PDG file " << std::endl;
        }
      }
    };
};

#endif  //Geometry_VeryForwardRPTopology_RPHepPDTWrapper
