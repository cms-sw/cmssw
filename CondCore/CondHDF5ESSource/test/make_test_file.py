import h5py
import numpy as np

with h5py.File('test.h5', 'w') as h5file:
    h5file.attrs["at"] = "fileAt".encode("ascii")
    AGroup = h5file.create_group("Agroup")
    AGroup.attrs["b_at"] = "groupAt".encode("ascii")
    dset = AGroup.create_dataset("byte_array", data=np.array([1],dtype='b'))
    dset.attrs["d_at"] = "dsetAt".encode("ascii")
    BGroup = AGroup.create_group("Bgroup")

    RefGroup = h5file.create_group("RefGroup")
    groupRefDS = RefGroup.create_dataset("groupRefs", data=[AGroup.ref, BGroup.ref], dtype=h5py.ref_dtype)
    dsetRefDS = RefGroup.create_dataset("dsetRefs", data=[dset.ref], dtype=h5py.ref_dtype)

    #syncValue
    syncValueType = np.dtype([("high", np.uint32),("low", np.uint32)])
    firstValues = ((1,0),(2,1),(0xFFFFFFFF,0xFFFFFFFF))
    first_np = np.empty(shape=(len(firstValues),), dtype=syncValueType)
    first_np['high'] = [ x[0] for x in firstValues]
    first_np['low'] = [ x[1] for x in firstValues]
    SyncGroup = h5file.create_group("SyncGroup")
    sDS = SyncGroup.create_dataset("sync", data=first_np)
