#include "RecoParticleFlow/Benchmark/interface/histo_manager.h"
#include "RecoParticleFlow/Benchmark/interface/histo.h"
#include <iostream>
#include <string.h>

using namespace std;


histo_manager::histo_manager()
{
this->var1_=new char(100);
this->var2_=new char(100);
this->var3_=new char(100);

}

histo_manager::~histo_manager()
{
}

void histo_manager::read_global_configfile()
{
	cout<<"================== READING CONF DATA ========================="<<endl;
	FILE *input=fopen("histos.dat","r");
	do
	{
		fscanf(input,"%s %s %s %d %d", this->var1_, this->var2_, this->var3_, &this->var4_, &this->var5_);
		if(strcmp(this->var1_,"EOF"))
		{
			hist_=new histo();
			cout<<this->var1_<<" "<<this->var2_<<" "<<this->var3_<<" "<<this->var4_<<" "<<this->var5_<<endl;
			this->hist_->setname(this->var1_);
			this->hist_->setbins(this->var4_);
			this->hist_->setmin(this->var4_);
			this->hist_->setmax(this->var5_);
			this->hstore_.push_back((*hist_));
		}
	}while(strcmp(this->var1_,"EOF"));
//	cout<<hstore_.size()<<endl;
	cout<<"==================== END READING CONF DATA ==================="<<endl<<endl;
}


int histo_manager::get_histosize()
{
	return this->hstore_.size();
}

void histo_manager::get_histo(int index, string *name, int *bins, int *min, int *max)
{
	//cout<<"Name1: "<<name<<endl;
	(*name)=(*this->hstore_[index].getname());
	(*bins)=this->hstore_[index].getbins();
	(*min)=this->hstore_[index].getmin();
	(*max)=this->hstore_[index].getmax();
	//cout<<"Name2: "<<name<<endl;
}

void read_user_configfile()
{

}

void bookhisto(string *name, int *bins, int *min, int *max)
{

}
