#!/usr/bin/env python3
import h5py
import numpy as np

with h5py.File('test.h5', 'w') as h5file:
    h5file.attrs["at"] = "fileAt".encode("ascii")
    AGroup = h5file.create_group("Agroup")
    AGroup.attrs["b_at"] = "groupAt".encode("ascii")
    dset = AGroup.create_dataset("byte_array", data=np.array([1],dtype='b'))
    dset.attrs["d_at"] = "dsetAt".encode("ascii")
    dset2 = AGroup.create_dataset("byte_array2", data=np.array([2,2],dtype='b'))
    dset3 = AGroup.create_dataset("byte_array3", data=np.array([3,3,3],dtype='b'))
    dset4 = AGroup.create_dataset("byte_array4", data=np.array([4,4,4,4],dtype='b'))
    BGroup = AGroup.create_group("Bgroup")

    RefGroup = h5file.create_group("RefGroup")
    groupRefDS = RefGroup.create_dataset("groupRefs", data=[AGroup.ref, BGroup.ref], dtype=h5py.ref_dtype)
    dsetRefDS = RefGroup.create_dataset("dsetRefs", data=[dset.ref], dtype=h5py.ref_dtype)
    dset2DRefDS = RefGroup.create_dataset("dset2DRefs", data=[[dset.ref,dset2.ref],[dset3.ref,dset4.ref]], dtype=h5py.ref_dtype)
    

    #syncValue
    syncValueType = np.dtype([("high", np.uint32),("low", np.uint32)])
    firstValues = ((1,0),(2,1),(0xFFFFFFFF,0xFFFFFFFF))
    first_np = np.empty(shape=(len(firstValues),), dtype=syncValueType)
    first_np['high'] = [ x[0] for x in firstValues]
    first_np['low'] = [ x[1] for x in firstValues]
    SyncGroup = h5file.create_group("SyncGroup")
    sDS = SyncGroup.create_dataset("sync", data=first_np)
