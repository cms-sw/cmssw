# T5 DNN working points (2026-09 retrain)

Model: `t5dnn_sel.pt (fork branch lst-t5dnn-train: analysis/DNN/t5dnn_retrain_2026-09/models); shipped table: 0.95`

Cut: pass = 1 - P(fake) > kWp[pt][eta]; a table with retention r keeps the fraction r of fully matched (pMatched > 0.95, i.e. all 10 hits from one simulated particle) T5s per bin. The fake column uses the standard 75% matching (t5_isFake). Held-out = val+test events of the training split (PU200 ttbar = pu, jets 0-499 = jet), un-selected T5 population (DNN cut off).

Bins: pt bin 0 = pt < 5 GeV, 1 = pt > 5 GeV; eta bins of 0.25 in |eta| of the first anchor hit (last open).

## Held-out kept fractions (kept/total)

### pu

| rule | fake | prompt | disp1-5 | disp5-25 | disp25-1e+09 |
|---|---|---|---|---|---|
| previous DNN (kWp98 on the old score) | 0.1727 | 0.8491 | 0.8341 | 0.8058 | 0.7147 |
| new_0.9 | 0.0209 | 0.6447 | 0.6367 | 0.6752 | 0.6633 |
| new_0.93 | 0.0304 | 0.6909 | 0.6889 | 0.7272 | 0.7170 |
| new_0.95 | 0.0423 | 0.7334 | 0.7340 | 0.7695 | 0.7596 |
| new_0.97 | 0.0664 | 0.7944 | 0.7947 | 0.8249 | 0.8166 |
| new_0.98 | 0.0912 | 0.8378 | 0.8360 | 0.8607 | 0.8568 |
| new_0.99 | 0.1468 | 0.8974 | 0.8950 | 0.9106 | 0.9072 |
| new_0.992 | 0.1682 | 0.9124 | 0.9095 | 0.9225 | 0.9188 |

### jet

| rule | fake | prompt | disp1-5 | disp5-25 | disp25-1e+09 |
|---|---|---|---|---|---|
| previous DNN (kWp98 on the old score) | 0.2752 | 0.6277 | 0.6990 | 0.6879 | 0.7078 |
| new_0.9 | 0.0014 | 0.3363 | 0.3807 | 0.3549 | 0.3119 |
| new_0.93 | 0.0026 | 0.3966 | 0.4422 | 0.4129 | 0.3683 |
| new_0.95 | 0.0046 | 0.4574 | 0.5028 | 0.4668 | 0.4215 |
| new_0.97 | 0.0096 | 0.5524 | 0.5954 | 0.5463 | 0.5051 |
| new_0.98 | 0.0157 | 0.6266 | 0.6612 | 0.6079 | 0.5669 |
| new_0.99 | 0.0369 | 0.7493 | 0.7795 | 0.7040 | 0.6642 |
| new_0.992 | 0.0418 | 0.7700 | 0.8021 | 0.7246 | 0.6860 |

## Tables

### 0.9

```cpp
HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
    {0.9553f, 0.9620f, 0.9731f, 0.9659f, 0.9017f, 0.9171f, 0.9508f, 0.9668f, 0.9816f, 0.9807f},
    {0.9917f, 0.9907f, 0.9938f, 0.9898f, 0.9690f, 0.9360f, 0.9463f, 0.9526f, 0.9575f, 0.9453f}};
```

Fully matched rows per bin (the quantile statistics): 142368,150310,210888,151340,221051,201767,284071,361798,271657,123139 / 3001,3159,3784,2871,3480,2952,3351,4256,3223,2110

### 0.93

```cpp
HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
    {0.9256f, 0.9328f, 0.9538f, 0.9467f, 0.8653f, 0.8862f, 0.9319f, 0.9539f, 0.9738f, 0.9720f},
    {0.9852f, 0.9843f, 0.9899f, 0.9826f, 0.9552f, 0.9134f, 0.9138f, 0.9278f, 0.9347f, 0.9160f}};
```

Fully matched rows per bin (the quantile statistics): 142368,150310,210888,151340,221051,201767,284071,361798,271657,123139 / 3001,3159,3784,2871,3480,2952,3351,4256,3223,2110

### 0.95

```cpp
HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
    {0.8838f, 0.8933f, 0.9270f, 0.9192f, 0.8223f, 0.8524f, 0.9099f, 0.9383f, 0.9640f, 0.9608f},
    {0.9761f, 0.9744f, 0.9855f, 0.9765f, 0.9327f, 0.8803f, 0.8803f, 0.8978f, 0.9049f, 0.8822f}};
```

Fully matched rows per bin (the quantile statistics): 142368,150310,210888,151340,221051,201767,284071,361798,271657,123139 / 3001,3159,3784,2871,3480,2952,3351,4256,3223,2110

### 0.97

```cpp
HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
    {0.7961f, 0.8094f, 0.8684f, 0.8648f, 0.7420f, 0.7894f, 0.8683f, 0.9068f, 0.9444f, 0.9356f},
    {0.9528f, 0.9531f, 0.9747f, 0.9623f, 0.8829f, 0.8174f, 0.8161f, 0.8220f, 0.8538f, 0.7899f}};
```

Fully matched rows per bin (the quantile statistics): 142368,150310,210888,151340,221051,201767,284071,361798,271657,123139 / 3001,3159,3784,2871,3480,2952,3351,4256,3223,2110

### 0.98

```cpp
HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
    {0.7011f, 0.7271f, 0.8038f, 0.8049f, 0.6671f, 0.7305f, 0.8244f, 0.8723f, 0.9229f, 0.9086f},
    {0.9281f, 0.9345f, 0.9557f, 0.9402f, 0.8321f, 0.7069f, 0.7669f, 0.7378f, 0.8198f, 0.6664f}};
```

Fully matched rows per bin (the quantile statistics): 142368,150310,210888,151340,221051,201767,284071,361798,271657,123139 / 3001,3159,3784,2871,3480,2952,3351,4256,3223,2110

### 0.99

```cpp
HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
    {0.5158f, 0.5482f, 0.6421f, 0.6651f, 0.5309f, 0.6171f, 0.7343f, 0.7930f, 0.8679f, 0.8394f},
    {0.8521f, 0.8764f, 0.9106f, 0.8996f, 0.5076f, 0.5482f, 0.6789f, 0.6357f, 0.6518f, 0.4586f}};
```

Fully matched rows per bin (the quantile statistics): 142368,150310,210888,151340,221051,201767,284071,361798,271657,123139 / 3001,3159,3784,2871,3480,2952,3351,4256,3223,2110

### 0.992

```cpp
HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
    {0.4479f, 0.4844f, 0.5868f, 0.6181f, 0.4904f, 0.5828f, 0.7011f, 0.7590f, 0.8466f, 0.8129f},
    {0.8438f, 0.8579f, 0.8944f, 0.8867f, 0.4820f, 0.4591f, 0.6576f, 0.5655f, 0.6297f, 0.4283f}};
```

Fully matched rows per bin (the quantile statistics): 142368,150310,210888,151340,221051,201767,284071,361798,271657,123139 / 3001,3159,3784,2871,3480,2952,3351,4256,3223,2110

