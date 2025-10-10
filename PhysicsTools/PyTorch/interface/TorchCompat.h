#ifndef PhysicsTools_PyTorch_interface_TorchCompat_h
#define PhysicsTools_PyTorch_interface_TorchCompat_h

// TODO: find a better way to resolve TorchLib and ROOT's ClassDef macro clash.
// See: https://root-forum.cern.ch/t/use-of-torch-model-inside-root-dataframe-class-functor/62797
#ifdef ClassDef
#undef ClassDef
#endif

#include <torch/script.h>
#include <torch/torch.h>

#endif  // PhysicsTools_PyTorch_interface_TorchCompat_h
