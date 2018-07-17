## Uploading Tags to ProdDB

Here is an example of a metadata file to upload a tag with *HLT synchronization*. The field *since* should be left *null*. 
The IOV will be automatically taken and will be next safe run at HLT.
```
{
    "destinationDatabase": "oracle://cms_orcon_prod/CMS_CONDITIONS", 
    "destinationTags": {
        "XMLFILE_CTPPS_Geometry_101YV3_hlt": {}
    }, 
    "inputTag": "XMLFILE_CTPPS_Geometry_2018_102YV1", 
    "since": null, 
    "userText": "2018 CTPPS geometry"
}
```

#### IMPORTANT NOTE:
This should be done only after the tag has been validated.

All uploads must be annonced on [AlCaDB hypernews](https://hypernews.cern.ch/HyperNews/CMS/get/calibrations.html)
