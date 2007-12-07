#ifndef _HISTO_MANAGER_
#define _HISTO_MANAGER_

#include <string.h>
#include <vector>

#include "RecoParticleFlow/Benchmark/interface/histo.h"

class histo_manager{
	public:
		histo_manager();
		~histo_manager();
		void read_global_configfile();
		void read_user_configfile();
		int get_histosize();
		void get_histo(int, std::string*, int*, int*, int*);
		void bookhisto(std::string*, int*, int*, int*, int*);
	protected:
		char *var1_;
		char *var2_;
		char *var3_;
		int var4_;
		int var5_;
		std::vector<histo> hstore_;
		histo *hist_;
		
};
#endif
