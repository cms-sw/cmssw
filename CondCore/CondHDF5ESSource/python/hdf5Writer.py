import h5py
import zlib
import lzma
import numpy as np

#The file structure
#
# "format_version" - Attribute says which version of the file format was used
# "default_payload_compressor" - Attribute name of compressor used for the payloads
#
# "Records"- Group
#   <Record> - Group name is the EventSetup record name
#      "DataProducts" - Group
#         <data product> - Group name is the '<type>@<label>' combination
#            "type" - Attribute, the C++ canonical type name
#            "Payloads" - Group
#               <payload> - DataSet name is hash used in DB
#                  "memsize" = Attribute bytes needed after decompression
#                  "type" = Attribute the actual type stored (for polymorphism)
#      "Tags" - Group
#        <tag> - Group name is
#                   same as DB if only one data product is in the tag
#                   a hybrid name formed from the different DB tags it merged
#          "products" - Attribute, list of the data products used in the order they appear in "payload"
#          "time_type" - Attribute, either 'run_lumi' or 'time'
#          "db_tags" - Attribute the list of DB tags that were combine
#          "record" - Attribute name of the record to which the tag is associated (optimizes readback)
#          "first" - DataSet holds the beginning IOVSyncValue for the IOVs
#          "last" - DataSet holds the end IOVSyncValue for the IOVS
#          "payload" - DataSet references to the payloads for this IOV for each data product
#
# "GlobalTags" - Group
#   <global tag> - Group name is the global tag name
#      "Tags" - DataSet holds references to the tags


def writeTagImpl(tagsGroup, name, recName, time_type, IOV_payloads, payloadToRefs, productNames, originalTagNames):
    tagGroup = tagsGroup.create_group(name)
    tagGroup.attrs["time_type"] = time_type.encode("ascii")
    tagGroup.attrs["db_tags"] = [x.encode("ascii") for x in originalTagNames]
    tagGroup.attrs["record"] = recName.encode("ascii")
    tagGroup.attrs['products'] = [x.encode("ascii") for x in productNames]
    firstValues = [x[0] for x in IOV_payloads]
    lastValues = [x[1] for x in IOV_payloads]
    syncValueType = np.dtype([("high", np.uint32),("low", np.uint32)])
    first_np = np.empty(shape=(len(IOV_payloads),), dtype=syncValueType)
    first_np['high'] = [ x.high for x in firstValues]
    first_np['low'] = [ x.low for x in firstValues]
    last_np = np.empty(shape=(len(lastValues),), dtype=syncValueType)
    last_np['high'] = [ x.high for x in lastValues]
    last_np['low'] = [ x.low for x in lastValues]
    #tagGroup.create_dataset("first",data=np.array(firstValues), dtype=syncValueType)
    #tagGroup.create_dataset("last", data=np.array(lastValues),dtype=syncValueType)
    payloads = [ [ payloadToRefs[y] for y in x[2]] for x in IOV_payloads]
    compressor = None
    if len(first_np) > 100:
        compressor = 'gzip'
    tagGroup.create_dataset("first",data=first_np, compression = compressor)
    tagGroup.create_dataset("last",data=last_np, compression = compressor)
    tagGroup.create_dataset("payload", data=payloads, dtype=h5py.ref_dtype, compression = compressor)
    return tagGroup.ref

    
def writeTag(tagsGroup, time_type, IOV_payloads, payloadToRefs, originalTagNames, recName, productNames):
    name = originalTagNames[0]
    if len(originalTagNames) != 1:
        name = name+"@joined"
    return writeTagImpl(tagsGroup, name, recName, time_type, IOV_payloads, payloadToRefs, productNames, originalTagNames)

def writeH5File(fileName, globalTags, excludeRecords, includeRecords, tagReader, compressorName):
    #what are key lists??? They seem to hold objects of type 'cond::persistency::KeyList'
    # and have their own proxy type
    keyListRecords = set(["ExDwarfListRcd", "DTKeyedConfigListRcd", "DTKeyedConfigContainerRcd"])

    default_compressor_name = compressorName
    print(default_compressor_name)
    default_compressor = None
    if default_compressor_name == "zlib":
        default_compressor = zlib
    elif default_compressor_name == "lzma":
        default_compressor = lzma
    with h5py.File(fileName, 'w') as h5file:
        h5file.attrs["file_format"] = 1
        h5file.attrs["default_payload_compressor"] = default_compressor_name.encode("ascii")
        recordsGroup = h5file.create_group("Records")
        globalTagsGroup = h5file.create_group("GlobalTags")
        null_dataset = h5file.create_dataset("null_payload", data=np.array([], dtype='b') )
        tagGroupRefs = []
        
        for name in globalTags:
            gt = tagReader(name)
            for tag in gt.tags():
                rcd = tag.record()
                if rcd in keyListRecords:
                    continue
                if rcd in excludeRecords:
                    continue
                if includeRecords and (not rcd in includeRecords):
                    continue
                recordDataSize = 0
                
                payloadToRefs = { None: null_dataset.ref}
                
                recordGroup = recordsGroup.create_group(rcd)
                tagsGroup = recordGroup.create_group("Tags")
                dataProductsGroup = recordGroup.create_group("DataProducts")
                print("record: %s"%rcd)
                productNames = []
                for dataProduct in tag.dataProducts():
                    productNames.append(dataProduct.name())
                    dataProductGroup = dataProductsGroup.create_group(dataProduct.name())
                    dataProductGroup.attrs["type"] = dataProduct.objtype().encode("ascii")
                    payloadsGroup = dataProductGroup.create_group("Payloads")
                    print(" product: %s"%dataProduct.name())
                    for p_index, payload in enumerate(dataProduct.payloads()):
                        print("  %i payload: %s size: %i"%(p_index,payload.name(),len(payload.data())))
                        recordDataSize +=len(payload.data())
                        if default_compressor:
                            b = default_compressor.compress(payload.data())
                            if len(b) >= len(payload.data()):
                                #compressing isn't helping
                                b = payload.data()
                        else:
                            b = payload.data()
                        pl = payloadsGroup.create_dataset(payload.name(), data=np.frombuffer(b,dtype='b'))
                        pl.attrs["memsize"] = len(payload.data())
                        pl.attrs["type"] = payload.actualType()
                        payloadToRefs[payload.name()] = pl.ref
                        
                tagGroupRefs.append(writeTag(tagsGroup, tag.time_type(), tag.iovsNPayloadNames(), payloadToRefs, tag.originalTagNames(), rcd, productNames))
                print(" total size:",recordDataSize)
                recordDataSize = 0

            globalTagGroup = globalTagsGroup.create_group(name)
            globalTagGroup.create_dataset("Tags", data=tagGroupRefs, dtype=h5py.ref_dtype)
