#ifndef SISTRIPCLUSTER_CLASSES_H
#define SISTRIPCLUSTER_CLASSES_H

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/ContainerMask.h"
namespace DataFormats_SiStripCluster {
  struct dictionary2 {
    edm::Wrapper< SiStripCluster > dummy0;
    edm::Wrapper< std::vector<SiStripCluster>  > dummy1;

    edm::Wrapper< edmNew::DetSetVector<SiStripCluster> > dummy4_bis;
    edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > dummy_cm1;
    edm::Wrapper<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > > dummy_w_cm1;

    edmNew::DetSetVector<edm::Ref<edmNew::DetSetVector<SiStripCluster>,SiStripCluster,edmNew::DetSetVector<SiStripCluster>::FindForDetSetVector> > dumm_dtvr;
    edm::Wrapper<edmNew::DetSetVector<edm::Ref<edmNew::DetSetVector<SiStripCluster>,SiStripCluster,edmNew::DetSetVector<SiStripCluster>::FindForDetSetVector> > > dumm_dtvr_w;

    edm::Ref<edmNew::DetSetVector<SiStripCluster>, SiStripCluster, edmNew::DetSetVector<SiStripCluster>::FindForDetSetVector > refNew;
  };
}


#endif // SISTRIPCLUSTER_CLASSES_H
