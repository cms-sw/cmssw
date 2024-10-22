
M_X = 1.
M_Y = 2.5
M_T = .2

PIX_D_W = 10.
PIX_L_W = 20.
PIX_L_H = 10.

STR_D_W = 10.
STR_TID_D_W = 5.5
STR_L_W = 24.
STR_L_H = 12.

TkMap_GEO = {
    'pixel' : {
        'BPIX': {
            'L1': {
                'file': 'Relative_TPolyLine_Pixel_phase1_BPIX_L1.txt',
                'x_off': -2*PIX_L_W - 2*M_X,
                'y_off': -PIX_L_H, 
                'x_scale': PIX_L_W,
                'y_scale': PIX_L_H,
            },
            'L2': {
                'file': 'Relative_TPolyLine_Pixel_phase1_BPIX_L2.txt',
                'x_off': -PIX_L_W - M_X,
                'y_off': -PIX_L_H, 
                'x_scale': PIX_L_W,
                'y_scale': PIX_L_H,
            },
            'L3': {
                'file': 'Relative_TPolyLine_Pixel_phase1_BPIX_L3.txt',
                'x_off': -2*PIX_L_W - 2*M_X,
                'y_off': M_Y,
                'x_scale': PIX_L_W,
                'y_scale': PIX_L_H,
            },
            'L4': {
                'file': 'Relative_TPolyLine_Pixel_phase1_BPIX_L4.txt',
                'x_off': -PIX_L_W - M_X,
                'y_off': M_Y,
                'x_scale': PIX_L_W,
                'y_scale': PIX_L_H,
            },
        },
        'FPIX-': {
            '-1': {
                'file': 'Relative_TPolyLine_Pixel_phase1_FPIX_-1.txt',
                'x_off': -2*PIX_L_W - 2*M_X,
                'y_off': -PIX_L_H - M_Y - PIX_D_W, 
                'x_scale': PIX_D_W,
                'y_scale': PIX_D_W,
            },
            '-2': {
                'file': 'Relative_TPolyLine_Pixel_phase1_FPIX_-2.txt',
                'x_off': -PIX_L_W - M_X - PIX_L_W/4. - M_X/2.,
                'y_off': -PIX_L_H - M_Y - PIX_D_W, 
                'x_scale': PIX_D_W,
                'y_scale': PIX_D_W,
            },
            '-3': {
                'file': 'Relative_TPolyLine_Pixel_phase1_FPIX_-3.txt',
                'x_off': -PIX_D_W - M_X,
                'y_off': -PIX_L_H - M_Y - PIX_D_W, 
                'x_scale': PIX_D_W,
                'y_scale': PIX_D_W,
            },
        },
        'FPIX+': {
            '+1': {
                'file': 'Relative_TPolyLine_Pixel_phase1_FPIX_+1.txt',
                'x_off': -2*PIX_L_W - 2*M_X,
                'y_off': PIX_L_H + 2*M_Y, 
                'x_scale': PIX_D_W,
                'y_scale': PIX_D_W,
            },
            '+2': {
                'file': 'Relative_TPolyLine_Pixel_phase1_FPIX_+2.txt',
                'x_off': -PIX_L_W - M_X - PIX_L_W/4. - M_X/2.,
                'y_off': PIX_L_H + 2*M_Y, 
                'x_scale': PIX_D_W,
                'y_scale': PIX_D_W,
            },
            '+3': {
                'file': 'Relative_TPolyLine_Pixel_phase1_FPIX_+3.txt',
                'x_off': -PIX_D_W- M_X,
                'y_off': PIX_L_H + 2*M_Y, 
                'x_scale': PIX_D_W,
                'y_scale': PIX_D_W,
            },
        }, 
    },
    'strips': {
        'TIB': {
            'L1': {
                'file': 'Relative_TPolyLine_Strips_TIB_L1.txt',
                'x_off': M_X,
                'y_off': -STR_L_H,
                'x_scale': STR_L_W,
                'y_scale': STR_L_H,
            },
            'L2': {
                'file': 'Relative_TPolyLine_Strips_TIB_L2.txt',
                'x_off': 2*M_X + STR_L_W,
                'y_off': -STR_L_H,
                'x_scale': STR_L_W,
                'y_scale': STR_L_H,
            },
            'L3': {
                'file': 'Relative_TPolyLine_Strips_TIB_L3.txt',
                'x_off': M_X,
                'y_off': M_Y,
                'x_scale': STR_L_W,
                'y_scale': STR_L_H,
            },
            'L4': {
                'file': 'Relative_TPolyLine_Strips_TIB_L4.txt',
                'x_off': 2*M_X + STR_L_W,
                'y_off': M_Y,
                'x_scale': STR_L_W,
                'y_scale': STR_L_H,
            },
        },
        'TOB': {
            'L1': {
                'file': 'Relative_TPolyLine_Strips_TOB_L1.txt',
                'x_off': 3*M_X + 2*STR_L_W,
                'y_off': -STR_L_H - M_Y - STR_L_H/2. + M_Y/2.,
                'x_scale': STR_L_W,
                'y_scale': STR_L_H,
            },
            'L2': {
                'file': 'Relative_TPolyLine_Strips_TOB_L2.txt',
                'x_off': 4*M_X + 3*STR_L_W,
                'y_off': -STR_L_H - M_Y - STR_L_H/2. + M_Y/2.,
                'x_scale': STR_L_W,
                'y_scale': STR_L_H,
            },
            'L3': {
                'file': 'Relative_TPolyLine_Strips_TOB_L3.txt',
                'x_off': 3*M_X + 2*STR_L_W,
                'y_off': - STR_L_H/2. + M_Y/2.,
                'x_scale': STR_L_W,
                'y_scale': STR_L_H,
            },
            'L4': {
                'file': 'Relative_TPolyLine_Strips_TOB_L4.txt',
                'x_off': 4*M_X + 3*STR_L_W,
                'y_off': - STR_L_H/2. + M_Y/2.,
                'x_scale': STR_L_W,
                'y_scale': STR_L_H,
            },
            'L5': {
                'file': 'Relative_TPolyLine_Strips_TOB_L5.txt',
                'x_off': 3*M_X + 2*STR_L_W,
                'y_off': STR_L_H/2. + M_Y + M_Y/2.,
                'x_scale': STR_L_W,
                'y_scale': STR_L_H,
            },
            'L6': {
                'file': 'Relative_TPolyLine_Strips_TOB_L6.txt',
                'x_off': 4*M_X + 3*STR_L_W,
                'y_off': STR_L_H/2. + M_Y + M_Y/2.,
                'x_scale': STR_L_W,
                'y_scale': STR_L_H,
            },
        },
        'TID-': {
            '-1': {
                'file': 'Relative_TPolyLine_Strips_TID_-1.txt',
                'x_off': M_X + (STR_L_W + M_X)/2.,
                'y_off': - STR_L_H - M_Y - STR_TID_D_W,
                'x_scale': STR_TID_D_W,
                'y_scale': STR_TID_D_W,
            },
            '-2': {
                'file': 'Relative_TPolyLine_Strips_TID_-2.txt',
                'x_off': M_X + STR_L_W/2. - STR_TID_D_W/2. + (STR_L_W + M_X)/2.,
                'y_off': - STR_L_H - M_Y - STR_TID_D_W,
                'x_scale': STR_TID_D_W,
                'y_scale': STR_TID_D_W,
            },
            '-3': {
                'file': 'Relative_TPolyLine_Strips_TID_-3.txt',
                'x_off': M_X + STR_L_W - STR_TID_D_W + (STR_L_W + M_X)/2.,
                'y_off': - STR_L_H - M_Y - STR_TID_D_W,
                'x_scale': STR_TID_D_W,
                'y_scale': STR_TID_D_W,
            },
        },
        'TID+': {
            '+1': {
                'file': 'Relative_TPolyLine_Strips_TID_+1.txt',
                'x_off': M_X + (STR_L_W + M_X)/2.,
                'y_off': STR_L_H +2*M_Y,
                'x_scale': STR_TID_D_W,
                'y_scale': STR_TID_D_W,
            },
            '+2': {
                'file': 'Relative_TPolyLine_Strips_TID_+2.txt',
                'x_off': M_X + STR_L_W/2. - STR_TID_D_W/2. + (STR_L_W + M_X)/2.,
                'y_off': STR_L_H +2*M_Y,
                'x_scale': STR_TID_D_W,
                'y_scale': STR_TID_D_W,
            },
            '+3': {
                'file': 'Relative_TPolyLine_Strips_TID_+3.txt',
                'x_off': M_X + STR_L_W - STR_TID_D_W + (STR_L_W + M_X)/2.,
                'y_off': STR_L_H +2*M_Y,
                'x_scale': STR_TID_D_W,
                'y_scale': STR_TID_D_W,
            },
        },
        'TEC-': {
            '-1': {
                'file': 'Relative_TPolyLine_Strips_TEC_-1.txt',
                'x_off': M_X,
                'y_off': -STR_L_H - M_Y - STR_L_H/2. + M_Y/2. - M_Y - STR_D_W,
                'x_scale': STR_D_W,
                'y_scale': STR_D_W,
            },
            '-2': {
                'file': 'Relative_TPolyLine_Strips_TEC_-2.txt',
                'x_off': M_X +((4*STR_L_W + 3*M_X - 9*STR_D_W)/8. + STR_D_W),
                'y_off': -STR_L_H - M_Y - STR_L_H/2. + M_Y/2. - M_Y - STR_D_W,
                'x_scale': STR_D_W,
                'y_scale': STR_D_W,
            },
            '-3': {
                'file': 'Relative_TPolyLine_Strips_TEC_-3.txt',
                'x_off': M_X + 2*((4*STR_L_W + 3*M_X - 9*STR_D_W)/8. + STR_D_W),
                'y_off': -STR_L_H - M_Y - STR_L_H/2. + M_Y/2. - M_Y - STR_D_W,
                'x_scale': STR_D_W,
                'y_scale': STR_D_W,
            },
            '-4': {
                'file': 'Relative_TPolyLine_Strips_TEC_-4.txt',
                'x_off': M_X + 3*((4*STR_L_W + 3*M_X - 9*STR_D_W)/8. + STR_D_W),
                'y_off': -STR_L_H - M_Y - STR_L_H/2. + M_Y/2. - M_Y - STR_D_W,
                'x_scale': STR_D_W,
                'y_scale': STR_D_W,
            },
            '-5': {
                'file': 'Relative_TPolyLine_Strips_TEC_-5.txt',
                'x_off': M_X + 4*((4*STR_L_W + 3*M_X - 9*STR_D_W)/8. + STR_D_W),
                'y_off': -STR_L_H - M_Y - STR_L_H/2. + M_Y/2. - M_Y - STR_D_W,
                'x_scale': STR_D_W,
                'y_scale': STR_D_W,
            },
            '-6': {
                'file': 'Relative_TPolyLine_Strips_TEC_-6.txt',
                'x_off': M_X + 5*((4*STR_L_W + 3*M_X - 9*STR_D_W)/8. + STR_D_W),
                'y_off': -STR_L_H - M_Y - STR_L_H/2. + M_Y/2. - M_Y - STR_D_W,
                'x_scale': STR_D_W,
                'y_scale': STR_D_W,
            },
            '-7': {
                'file': 'Relative_TPolyLine_Strips_TEC_-7.txt',
                'x_off': M_X + 6*((4*STR_L_W + 3*M_X - 9*STR_D_W)/8. + STR_D_W),
                'y_off': -STR_L_H - M_Y - STR_L_H/2. + M_Y/2. - M_Y - STR_D_W,
                'x_scale': STR_D_W,
                'y_scale': STR_D_W,
            },
            '-8': {
                'file': 'Relative_TPolyLine_Strips_TEC_-8.txt',
                'x_off': M_X + 7*((4*STR_L_W + 3*M_X - 9*STR_D_W)/8. + STR_D_W),
                'y_off': -STR_L_H - M_Y - STR_L_H/2. + M_Y/2. - M_Y - STR_D_W,
                'x_scale': STR_D_W,
                'y_scale': STR_D_W,
            },
            '-9': {
                'file': 'Relative_TPolyLine_Strips_TEC_-9.txt',
                'x_off': M_X + 8*((4*STR_L_W + 3*M_X - 9*STR_D_W)/8. + STR_D_W),
                'y_off': -STR_L_H - M_Y - STR_L_H/2. + M_Y/2. - M_Y - STR_D_W,
                'x_scale': STR_D_W,
                'y_scale': STR_D_W,
            },
        },
        'TEC+': {
            '+1': {
                'file': 'Relative_TPolyLine_Strips_TEC_+1.txt',
                'x_off': M_X,
                'y_off': STR_L_H + STR_L_H/2. + M_Y/2. + 2*M_Y,
                'x_scale': STR_D_W,
                'y_scale': STR_D_W,
            },
            '+2': {
                'file': 'Relative_TPolyLine_Strips_TEC_+2.txt',
                'x_off': M_X + ((4*STR_L_W + 3*M_X - 9*STR_D_W)/8. + STR_D_W),
                'y_off': STR_L_H + STR_L_H/2. + M_Y/2. + 2*M_Y,
                'x_scale': STR_D_W,
                'y_scale': STR_D_W,
            },
            '+3': {
                'file': 'Relative_TPolyLine_Strips_TEC_+3.txt',
                'x_off': M_X + 2*((4*STR_L_W + 3*M_X - 9*STR_D_W)/8. + STR_D_W),
                'y_off': STR_L_H + STR_L_H/2. + M_Y/2. + 2*M_Y,
                'x_scale': STR_D_W,
                'y_scale': STR_D_W,
            },
            '+4': {
                'file': 'Relative_TPolyLine_Strips_TEC_+4.txt',
                'x_off': M_X + 3*((4*STR_L_W + 3*M_X - 9*STR_D_W)/8. + STR_D_W),
                'y_off': STR_L_H + STR_L_H/2. + M_Y/2. + 2*M_Y,
                'x_scale': STR_D_W,
                'y_scale': STR_D_W,
            },
            '+5': {
                'file': 'Relative_TPolyLine_Strips_TEC_+5.txt',
                'x_off': M_X + 4*((4*STR_L_W + 3*M_X - 9*STR_D_W)/8. + STR_D_W),
                'y_off': STR_L_H + STR_L_H/2. + M_Y/2. + 2*M_Y,
                'x_scale': STR_D_W,
                'y_scale': STR_D_W,
            },
            '+6': {
                'file': 'Relative_TPolyLine_Strips_TEC_+6.txt',
                'x_off': M_X + 5*((4*STR_L_W + 3*M_X - 9*STR_D_W)/8. + STR_D_W),
                'y_off': STR_L_H + STR_L_H/2. + M_Y/2. + 2*M_Y,
                'x_scale': STR_D_W,
                'y_scale': STR_D_W,
            },
            '+7': {
                'file': 'Relative_TPolyLine_Strips_TEC_+7.txt',
                'x_off': M_X + 6*((4*STR_L_W + 3*M_X - 9*STR_D_W)/8. + STR_D_W),
                'y_off': STR_L_H + STR_L_H/2. + M_Y/2. + 2*M_Y,
                'x_scale': STR_D_W,
                'y_scale': STR_D_W,
            },
            '+8': {
                'file': 'Relative_TPolyLine_Strips_TEC_+8.txt',
                'x_off': M_X + 7*((4*STR_L_W + 3*M_X - 9*STR_D_W)/8. + STR_D_W),
                'y_off': STR_L_H + STR_L_H/2. + M_Y/2. + 2*M_Y,
                'x_scale': STR_D_W,
                'y_scale': STR_D_W,
            },
            '+9': {
                'file': 'Relative_TPolyLine_Strips_TEC_+9.txt',
                'x_off': M_X + 8*((4*STR_L_W + 3*M_X - 9*STR_D_W)/8. + STR_D_W),
                'y_off': STR_L_H + STR_L_H/2. + M_Y/2. + 2*M_Y,
                'x_scale': STR_D_W,
                'y_scale': STR_D_W,
            },
        },
    },
}

# Add text
for layer in TkMap_GEO['pixel']['BPIX']:
    TkMap_GEO['pixel']['BPIX'][layer]['latex'] = {
        'text': 'BPIX '+layer,
        'x': TkMap_GEO['pixel']['BPIX'][layer]['x_off'],
        'y': TkMap_GEO['pixel']['BPIX'][layer]['y_off'] + PIX_L_H + M_T,
        'alignment': 11,
    }

for z in ['-', '+']:
    TkMap_GEO['pixel']['FPIX'+z]['latex'] = {
        'text': 'FPIX',
        'x': TkMap_GEO['pixel']['FPIX'+z][z+'1']['x_off'],
        'y': TkMap_GEO['pixel']['FPIX'+z][z+'1']['y_off'] + PIX_D_W + M_T,
        'alignment': 11,
    }
    for disc in TkMap_GEO['pixel']['FPIX'+z]:
        if disc == 'latex': continue
        TkMap_GEO['pixel']['FPIX'+z][disc]['latex'] = {
            'text': disc,
            'x': TkMap_GEO['pixel']['FPIX'+z][disc]['x_off'] + PIX_D_W,
            'y': TkMap_GEO['pixel']['FPIX'+z][disc]['y_off'] + PIX_D_W,
            #'alignment': 33,
            'alignment': 23,
        }
for det in ['TIB', 'TOB']:
    for layer in TkMap_GEO['strips'][det]:
        TkMap_GEO['strips'][det][layer]['latex'] = {
            'text': det+' '+layer,
            'x': TkMap_GEO['strips'][det][layer]['x_off'],
            'y': TkMap_GEO['strips'][det][layer]['y_off'] + STR_L_H + M_T,
            'alignment': 11,
        }

for z in ['-', '+']:
    TkMap_GEO['strips']['TEC'+z]['latex'] = {
        'text': 'TEC',
        'x': TkMap_GEO['strips']['TEC'+z][z+'1']['x_off'],
        'y': TkMap_GEO['strips']['TEC'+z][z+'1']['y_off'] + STR_D_W + M_T,
        'alignment': 11,
    }
    for disc in TkMap_GEO['strips']['TEC'+z]:
        if disc == 'latex': continue
        TkMap_GEO['strips']['TEC'+z][disc]['latex'] = {
            'text': disc,
            'x': TkMap_GEO['strips']['TEC'+z][disc]['x_off'] + STR_D_W,
            'y': TkMap_GEO['strips']['TEC'+z][disc]['y_off'] + STR_D_W,
            #'alignment': 33,
            'alignment': 23,
        }

# TID
for z in ['-', '+']:
    TkMap_GEO['strips']['TID'+z]['latex'] = {
        'text': 'TID',
        #'x': TkMap_GEO['strips']['TID'+z][z+'1']['x_off'],
        'x': TkMap_GEO['strips']['TID'+z][z+'1']['x_off'] - 2*M_T,
        #'y': TkMap_GEO['strips']['TID'+z][z+'1']['y_off'] + STR_TID_D_W + M_T,
        'y': TkMap_GEO['strips']['TID'+z][z+'1']['y_off'] + STR_TID_D_W/2.,
        #'alignment': 11,
        'alignment': 32,
    }
    for disc in TkMap_GEO['strips']['TID'+z]:
        if disc == 'latex': continue
        TkMap_GEO['strips']['TID'+z][disc]['latex'] = {
            'text': disc,
            'x': TkMap_GEO['strips']['TID'+z][disc]['x_off'] + STR_TID_D_W,
            'y': TkMap_GEO['strips']['TID'+z][disc]['y_off'] + STR_TID_D_W,
            'alignment': 13,
        }

def read_TPLfile(file_name):
    o_file = open(file_name, 'r')
    lines = o_file.readlines()
    o_file.close()

    TPL_dict = {}
    for line in lines:
        if '#' in line: continue
        splt_line = line.replace('\n', '').split(' ')
        det_id = int(splt_line[0])
        x = []
        y = []
        for idx,coo in enumerate(splt_line[1:]):
            #print(coo)
            try:
                val = float(coo)
                if (idx%2) == 0: 
                    y.append(val)
                else: 
                    x.append(val)
            except ValueError:
                continue
        TPL_dict[det_id] = {}
        TPL_dict[det_id]['x'] = x
        TPL_dict[det_id]['y'] = y
    return TPL_dict
        
    

if __name__ == '__main__':
    import ROOT
    from array import array
    #TPL_dict = read_TPLfile('Relative_TPolyLine_Pixel_phase1_BPIX_L1.txt')
    #print(TPL_dict)

    def test_draw(w, h):
        canvas = ROOT.TCanvas('canvas', 'detector canvas', w, h)
        
        # Load all modules in absolute positions
        # Find max x and y to rescale later for window
        all_modules = {}
        all_text = {}
        x_max = -9999.
        y_max = -9999.
        x_min = 9999.
        y_min = 9999.
        for det in TkMap_GEO:
            for sub in TkMap_GEO[det]:
                for part in TkMap_GEO[det][sub]:
                    if part == 'latex':
                        all_text[det+'_'+sub] = TkMap_GEO[det][sub][part]
                        continue
                    if 'latex' in TkMap_GEO[det][sub][part]:
                        all_text[det+'_'+sub+'_'+part] = TkMap_GEO[det][sub][part]['latex']
                    TPL_dict = read_TPLfile(TkMap_GEO[det][sub][part]['file'])
                    for module in TPL_dict:
                        x_canv = []
                        y_canv = []
                        for idx in range(len(TPL_dict[module]['x'])):
                            x_canv.append(TPL_dict[module]['x'][idx]*TkMap_GEO[det][sub][part]['x_scale'] + TkMap_GEO[det][sub][part]['x_off'])
                            y_canv.append(TPL_dict[module]['y'][idx]*TkMap_GEO[det][sub][part]['y_scale'] + TkMap_GEO[det][sub][part]['y_off'])
                        if max(x_canv) > x_max: x_max = max(x_canv)
                        if max(y_canv) > y_max: y_max = max(y_canv)
                        if min(x_canv) < x_min: x_min = min(x_canv)
                        if min(y_canv) < y_min: y_min = min(y_canv)
                        TPL_dict[module]['x'] = x_canv
                        TPL_dict[module]['y'] = y_canv
                    all_modules.update(TPL_dict)

        r_margin = 1
        l_margin = 1
        t_margin = 1
        b_margin = 1

        x_max += r_margin
        x_min -= l_margin
        y_max += t_margin
        y_min -= b_margin

        x_range = x_max - x_min
        y_range = y_max - y_min
        if (x_range + 0.)/(w + 0.) > (y_range + 0.)/(h + 0.):
            x_scale = x_range
            y_scale = (h + 0.)/(w + 0.)*x_range
        else:
            y_scale = y_range
            x_scale = (w + 0.)/(h + 0.)*y_range
        TPL = {} 
        for module in all_modules:
            x = array('d', [])
            y = array('d', [])
            for idx in range(len(all_modules[module]['x'])):
                x.append((all_modules[module]['x'][idx] - x_min + 0.)/(x_scale + 0.))
                y.append((all_modules[module]['y'][idx] - y_min + 0.)/(y_scale + 0.))
            # Begin point is end point
            x.append((all_modules[module]['x'][0] - x_min + 0.)/(x_scale + 0.))
            y.append((all_modules[module]['y'][0] - y_min + 0.)/(y_scale + 0.))
            #print(x, y)
            TPL[module] = ROOT.TPolyLine(len(x), x, y) 
            #TPL[module].SetFillColor(1)
            TPL[module].SetLineColor(1)
            TPL[module].Draw('f')
            TPL[module].Draw()
        TL = ROOT.TLatex()
        TL.SetTextSize(0.025)
        for key in all_text:
            x = (all_text[key]['x'] - x_min + 0.)/(x_scale + 0.)
            y = (all_text[key]['y'] - y_min + 0.)/(y_scale + 0.)
            TL.SetTextAlign(all_text[key]['alignment'])
            TL.DrawLatex(x, y, all_text[key]['text'])
        #TL.SetTextSize(0.025)
        #TL.SetTextAlign(11)
        #TL.DrawLatex(0.1, 0.1, 'bottom')
        #TL.SetTextAlign(13)
        #TL.DrawLatex(0.1, 0.1, 'top')
        canvas.Update() 
        raw_input('exit')
   
    test_draw(1500, 750)
    #test_draw(125, 500)
    #test_draw(500, 500)
    #raw_input('exit')
 
        
