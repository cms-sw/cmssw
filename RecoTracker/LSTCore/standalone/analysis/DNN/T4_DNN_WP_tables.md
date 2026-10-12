# T4 DNN working points (2026-09 retrain)

Model: `t4dnn_fullNoTier.pt (analysis/DNN/t4dnn_retrain_2026-09/models on the training branch); shipped table: pdisp0.9`

Cut: pass = P(displaced) > kWp[pt][eta]; pdisp<r> keeps the fraction r of fully matched (pMatched > 0.95, i.e. all 8 hits from one simulated particle) displaced (vxy >= 1 cm) T4s per bin; pfake<r> (1 - P(fake), all fully matched T4s) is for comparison only. The fake column uses the standard 75% matching (t4_isFake). Held-out = the val+test events (40% of each sample; 60/20/20 train/val/test split by event) of PU200 ttbar (pu) and QCD jets (jet), un-selected T4 population (DNN cut off), built behind the retrained T5 DNN.

Bins: pt bin 0 = pt < 5 GeV, 1 = pt > 5 GeV; eta bins of 0.25 in |eta| of the first anchor hit (last open).

## Held-out kept fractions (kept/total)

### pu

| rule | fake | prompt<0.1 | disp0.1-1 | disp1-5 | disp5-25 | disp25-1e+09 |
|---|---|---|---|---|---|---|
| previous DNN (kWp_displaced AND kWp_fake, 25 eta bins) | 0.1249 | 0.2841 | 0.3981 | 0.6833 | 0.7066 | 0.7870 |
| pfake0.8 | 0.0467 | 0.6781 | 0.7135 | 0.7534 | 0.6562 | 0.7330 |
| pdisp0.8 | 0.0507 | 0.1278 | 0.1865 | 0.5846 | 0.7150 | 0.8589 |
| pfake0.9 | 0.1045 | 0.8317 | 0.8577 | 0.8834 | 0.7898 | 0.8525 |
| pdisp0.9 | 0.1079 | 0.3176 | 0.4154 | 0.7539 | 0.8434 | 0.9315 |
| pfake0.95 | 0.1754 | 0.9182 | 0.9269 | 0.9372 | 0.8742 | 0.9204 |
| pdisp0.95 | 0.1853 | 0.5769 | 0.6538 | 0.8673 | 0.9132 | 0.9692 |
| pfake0.97 | 0.2419 | 0.9510 | 0.9615 | 0.9630 | 0.9150 | 0.9485 |
| pdisp0.97 | 0.2404 | 0.7287 | 0.7827 | 0.9167 | 0.9448 | 0.9813 |
| pfake0.98 | 0.2944 | 0.9666 | 0.9712 | 0.9768 | 0.9424 | 0.9624 |
| pdisp0.98 | 0.2851 | 0.8086 | 0.8577 | 0.9489 | 0.9589 | 0.9859 |
| pfake0.99 | 0.3826 | 0.9825 | 0.9769 | 0.9890 | 0.9654 | 0.9826 |
| pdisp0.99 | 0.3680 | 0.8900 | 0.9212 | 0.9740 | 0.9737 | 0.9916 |
| pfake0.995 | 0.4763 | 0.9912 | 0.9865 | 0.9945 | 0.9800 | 0.9921 |
| pdisp0.995 | 0.4452 | 0.9392 | 0.9462 | 0.9885 | 0.9827 | 0.9948 |

### jet

| rule | fake | prompt<0.1 | disp0.1-1 | disp1-5 | disp5-25 | disp25-1e+09 |
|---|---|---|---|---|---|---|
| previous DNN (kWp_displaced AND kWp_fake, 25 eta bins) | 0.1373 | 0.3793 | 0.5000 | 0.6353 | 0.6710 | 0.7510 |
| pfake0.8 | 0.0278 | 0.9069 | 0.8750 | 0.8118 | 0.8277 | 0.8169 |
| pdisp0.8 | 0.0593 | 0.2862 | 0.5000 | 0.6118 | 0.7311 | 0.8182 |
| pfake0.9 | 0.0548 | 0.9603 | 0.8750 | 0.8941 | 0.9217 | 0.8944 |
| pdisp0.9 | 0.0853 | 0.4397 | 0.6250 | 0.6824 | 0.8329 | 0.8886 |
| pfake0.95 | 0.0801 | 0.9707 | 1.0000 | 0.9529 | 0.9478 | 0.9302 |
| pdisp0.95 | 0.1231 | 0.6534 | 0.7500 | 0.8000 | 0.9060 | 0.9232 |
| pfake0.97 | 0.1007 | 0.9810 | 1.0000 | 0.9765 | 0.9634 | 0.9411 |
| pdisp0.97 | 0.1447 | 0.7690 | 0.7500 | 0.8706 | 0.9452 | 0.9558 |
| pfake0.98 | 0.1267 | 0.9879 | 1.0000 | 0.9765 | 0.9687 | 0.9469 |
| pdisp0.98 | 0.1626 | 0.8276 | 0.7500 | 0.8824 | 0.9556 | 0.9680 |
| pfake0.99 | 0.1459 | 0.9879 | 1.0000 | 0.9882 | 0.9739 | 0.9597 |
| pdisp0.99 | 0.1943 | 0.8793 | 0.8750 | 0.9412 | 0.9687 | 0.9776 |
| pfake0.995 | 0.1779 | 0.9897 | 1.0000 | 0.9882 | 0.9869 | 0.9635 |
| pdisp0.995 | 0.2252 | 0.9121 | 0.8750 | 0.9882 | 0.9765 | 0.9821 |

## Tables

### pfake0.8

```cpp
HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
    {0.5085f, 0.5482f, 0.6500f, 0.7314f, 0.5151f, 0.7821f, 0.8858f, 0.8733f, 0.9455f, 0.9866f},
    {0.2785f, 0.6644f, 0.9288f, 0.5752f, 0.6024f, 0.2593f, 0.8122f, 0.6709f, 0.8216f, 0.7178f}};
```

Fully matched rows per bin (the quantile statistics): 2186,2259,2458,1707,1877,1351,4208,5907,5574,16513 / 33,46,26,22,34,61,180,177,98,224

### pdisp0.8

```cpp
HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
    {0.2393f, 0.2877f, 0.5668f, 0.6725f, 0.3220f, 0.3243f, 0.5930f, 0.3855f, 0.3391f, 0.1156f},
    {0.0175f, 0.6660f, 0.7309f, 0.3451f, 0.2816f, 0.0944f, 0.3088f, 0.2046f, 0.6233f, 0.0551f}};
```

Fully matched rows per bin (the quantile statistics): 798,900,1201,1098,1066,782,2456,2867,1980,3311 / 19,20,15,11,19,19,73,76,44,47

### pfake0.9

```cpp
HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
    {0.2607f, 0.2346f, 0.3171f, 0.3847f, 0.3177f, 0.5645f, 0.7616f, 0.7113f, 0.8863f, 0.9613f},
    {0.0707f, 0.2841f, 0.7183f, 0.4809f, 0.4709f, 0.0999f, 0.6818f, 0.2982f, 0.7207f, 0.2260f}};
```

Fully matched rows per bin (the quantile statistics): 2186,2259,2458,1707,1877,1351,4208,5907,5574,16513 / 33,46,26,22,34,61,180,177,98,224

### pdisp0.9

```cpp
HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
    {0.0554f, 0.0925f, 0.2298f, 0.3026f, 0.1253f, 0.1596f, 0.3271f, 0.1529f, 0.1166f, 0.0593f},
    {0.0129f, 0.5345f, 0.2326f, 0.3337f, 0.1134f, 0.0325f, 0.1085f, 0.0919f, 0.3781f, 0.0417f}};
```

Fully matched rows per bin (the quantile statistics): 798,900,1201,1098,1066,782,2456,2867,1980,3311 / 19,20,15,11,19,19,73,76,44,47

### pfake0.95

```cpp
HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
    {0.1347f, 0.1077f, 0.1533f, 0.1896f, 0.1658f, 0.3459f, 0.5827f, 0.5140f, 0.7578f, 0.9074f},
    {0.0444f, 0.0922f, 0.4865f, 0.2805f, 0.3247f, 0.0368f, 0.5237f, 0.1677f, 0.4643f, 0.0803f}};
```

Fully matched rows per bin (the quantile statistics): 2186,2259,2458,1707,1877,1351,4208,5907,5574,16513 / 33,46,26,22,34,61,180,177,98,224

### pdisp0.95

```cpp
HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
    {0.0240f, 0.0309f, 0.0907f, 0.1360f, 0.0508f, 0.0593f, 0.1420f, 0.0680f, 0.0586f, 0.0382f},
    {0.0102f, 0.5033f, 0.0208f, 0.1808f, 0.0589f, 0.0206f, 0.0467f, 0.0483f, 0.3223f, 0.0363f}};
```

Fully matched rows per bin (the quantile statistics): 798,900,1201,1098,1066,782,2456,2867,1980,3311 / 19,20,15,11,19,19,73,76,44,47

### pfake0.97

```cpp
HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
    {0.0835f, 0.0508f, 0.0949f, 0.1269f, 0.0733f, 0.2294f, 0.4574f, 0.3698f, 0.6182f, 0.8397f},
    {0.0363f, 0.0363f, 0.4408f, 0.2028f, 0.1323f, 0.0303f, 0.1485f, 0.1125f, 0.2398f, 0.0536f}};
```

Fully matched rows per bin (the quantile statistics): 2186,2259,2458,1707,1877,1351,4208,5907,5574,16513 / 33,46,26,22,34,61,180,177,98,224

### pdisp0.97

```cpp
HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
    {0.0171f, 0.0153f, 0.0400f, 0.0806f, 0.0265f, 0.0366f, 0.0887f, 0.0514f, 0.0414f, 0.0284f},
    {0.0068f, 0.3147f, 0.0198f, 0.1196f, 0.0467f, 0.0184f, 0.0301f, 0.0313f, 0.3145f, 0.0320f}};
```

Fully matched rows per bin (the quantile statistics): 798,900,1201,1098,1066,782,2456,2867,1980,3311 / 19,20,15,11,19,19,73,76,44,47

### pfake0.98

```cpp
HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
    {0.0544f, 0.0304f, 0.0678f, 0.0892f, 0.0457f, 0.1600f, 0.3295f, 0.2865f, 0.5336f, 0.7503f},
    {0.0284f, 0.0077f, 0.4356f, 0.1646f, 0.1179f, 0.0196f, 0.0759f, 0.0922f, 0.1060f, 0.0475f}};
```

Fully matched rows per bin (the quantile statistics): 2186,2259,2458,1707,1877,1351,4208,5907,5574,16513 / 33,46,26,22,34,61,180,177,98,224

### pdisp0.98

```cpp
HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
    {0.0119f, 0.0103f, 0.0215f, 0.0557f, 0.0165f, 0.0295f, 0.0718f, 0.0393f, 0.0301f, 0.0246f},
    {0.0051f, 0.2205f, 0.0193f, 0.0890f, 0.0406f, 0.0173f, 0.0174f, 0.0241f, 0.3003f, 0.0284f}};
```

Fully matched rows per bin (the quantile statistics): 798,900,1201,1098,1066,782,2456,2867,1980,3311 / 19,20,15,11,19,19,73,76,44,47

### pfake0.99

```cpp
HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
    {0.0386f, 0.0108f, 0.0399f, 0.0595f, 0.0192f, 0.0827f, 0.1268f, 0.1391f, 0.3909f, 0.6022f},
    {0.0206f, 0.0067f, 0.4304f, 0.1264f, 0.1035f, 0.0148f, 0.0547f, 0.0483f, 0.0946f, 0.0405f}};
```

Fully matched rows per bin (the quantile statistics): 2186,2259,2458,1707,1877,1351,4208,5907,5574,16513 / 33,46,26,22,34,61,180,177,98,224

### pdisp0.99

```cpp
HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
    {0.0081f, 0.0060f, 0.0045f, 0.0407f, 0.0056f, 0.0187f, 0.0381f, 0.0269f, 0.0216f, 0.0203f},
    {0.0034f, 0.1262f, 0.0187f, 0.0584f, 0.0345f, 0.0162f, 0.0082f, 0.0218f, 0.2635f, 0.0217f}};
```

Fully matched rows per bin (the quantile statistics): 798,900,1201,1098,1066,782,2456,2867,1980,3311 / 19,20,15,11,19,19,73,76,44,47

### pfake0.995

```cpp
HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
    {0.0239f, 0.0011f, 0.0281f, 0.0434f, 0.0073f, 0.0305f, 0.0775f, 0.0832f, 0.2144f, 0.4387f},
    {0.0167f, 0.0062f, 0.4278f, 0.1073f, 0.0963f, 0.0138f, 0.0389f, 0.0389f, 0.0913f, 0.0106f}};
```

Fully matched rows per bin (the quantile statistics): 2186,2259,2458,1707,1877,1351,4208,5907,5574,16513 / 33,46,26,22,34,61,180,177,98,224

### pdisp0.995

```cpp
HOST_DEVICE_CONSTANT float kWp[kPtBins][kEtaBins] = {
    {0.0046f, 0.0024f, 0.0025f, 0.0327f, 0.0024f, 0.0071f, 0.0258f, 0.0165f, 0.0107f, 0.0165f},
    {0.0025f, 0.0790f, 0.0185f, 0.0431f, 0.0315f, 0.0156f, 0.0070f, 0.0208f, 0.2451f, 0.0183f}};
```

Fully matched rows per bin (the quantile statistics): 798,900,1201,1098,1066,782,2456,2867,1980,3311 / 19,20,15,11,19,19,73,76,44,47

