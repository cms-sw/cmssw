import matplotlib.pyplot as plt
import numpy as np
import json
import subprocess
import argparse
import mplhep as hep
import os
from matplotlib.ticker import ScalarFormatter
def getOrigRun(newRun, origX):
    for _newRun,run in origX:
        if newRun == _newRun: 
            return run
def unpack(i): 
    """unpack 64bit unsigned long long into 2 32bit unsigned int, return tuple (high,low) 
    """ 
    high=i>>32 
    low=i&0xFFFFFFFF 
    return(high,low) 

def plot(json_files, labels, output_file, f_ymin, f_ymax):
    CMS_Text_Size = 20
    CMS_Text = "Preliminary"
    lumitext = "pp collisions (13.6TeV)"
    xlabel = "Lumi Section Number"
    
    xlabel_font_size = 20
    ylabel_font_size = 20
    xlabel_location = "right"
    ylabel_location = "top"
    colors = ['blue',  'red', 'green','black', 'magenta', 'gray' , 'Cyan', 'darkorange', 'brown', 'purple']
    yvar = json_files[0].split("_")[1]
    ylabels = {"X":"X", "Y":"Y", "Z":"Z", "SigmaX":r"$\sigma_{X}$", "SigmaY":r"$\sigma_{Y}$", "SigmaZ":r"$\sigma_{Z}$", "dXdY":"dXdY", "dXdZ":"dXdZ", "dYdZ":"dYdZ"}
    ylabel = f"{ylabels[yvar]} [cm]"
  
 
    fig, ax = plt.subplots(figsize = (10, 10), dpi = 150)

    if hasattr(hep.cms, "lumitext"): # mplhep 0.4
        hep.cms.text(f"{CMS_Text}", exp = 'CMS',fontsize = CMS_Text_Size,ax=ax)
        hep.cms.lumitext(f"{lumitext}", fontsize = CMS_Text_Size,ax=ax)
    else:                            # mplhep >= 1.0
        hep.cms.text(text=f" {CMS_Text}", lumi=lumitext, ax=ax, fontsize=CMS_Text_Size)

    plt.style.use(hep.style.CMS)

    glob_y = []
    glob_x = []
    x_store = []
    y_store = []
    y_err_store = []
    for i,json_file in enumerate(json_files):
        # Load data
        with open(json_file, 'r') as f:
            input_data = json.load(f)

        # Extracting data
        x = [point['x'] for point in input_data['data']]
        y = [point['y'] for point in input_data['data']]
        glob_x += x
        glob_y += y
        y_err = [point['y_err'] for point in input_data['data']]
        
        # Store for further processing
        x_store.append(x)
        y_store.append(y)
        y_err_store.append(y_err)

    # find common x-axis
    minX = min(glob_x)
    maxX = max(glob_x)
    trueL = maxX-minX+1
    newX = []
    origX = []
    penalty = 0
    for run in range(minX,minX+trueL):
        x_not_in_all = True
        for x in x_store:
            if run in x: x_not_in_all = False
        if not x_not_in_all:
            newX.append(run-penalty)
            origX.append((run-penalty,run))
        else:
            penalty += 1

    # restore original y-values
    new_y_store = []
    new_y_err_store = []
    for ix,x in enumerate(x_store):
        new_y_store.append([])
        new_y_err_store.append([])
        newXpruned = []
        for run in newX:
            origRun = getOrigRun(run,origX)
            if origRun in x:
                newXpruned.append(run)
                new_y_store[ix].append(y_store[ix][x.index(origRun)]) 
                new_y_err_store[ix].append(y_err_store[ix][x.index(origRun)])    
        # Prep the data
        mean_y = np.mean(new_y_store[ix])
        std_dev_y = np.std(new_y_store[ix])
        _newXpruned = [x_val for x_val, y_val in zip(newXpruned, new_y_store[ix]) if abs(y_val - mean_y) <= 2 * std_dev_y]
        _newYpruned = [y_val for y_val in new_y_store[ix] if abs(y_val - mean_y) <= 2 * std_dev_y]
        _newYerrpruned = [y_err for y_val, y_err in zip(new_y_store[ix], new_y_err_store[ix]) if abs(y_val - mean_y) <= 2 * std_dev_y]
        newXpruned = _newXpruned
        new_y_store[ix] = _newYpruned
        new_y_err_store[ix] = _newYerrpruned
        
        # Prep linear fit
        coef = np.polyfit(newXpruned,new_y_store[ix],10)
        print(coef)
        newXpruned_new = [unpack(i)[1] for i in newXpruned]
        
        poly1d_fn = np.poly1d(coef)
        plt.tight_layout() 
        fig.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
        # Plot
        plt.errorbar(newXpruned_new,  new_y_store[ix], yerr=new_y_err_store[ix], color = colors[ix], fmt='o',  capsize=5, label=labels[ix], zorder=1)
        plt.plot(newXpruned_new, poly1d_fn(newXpruned), '--', color = colors[ix], label="lin. "+labels[ix])


    # Extract the annotations from the first file (assuming all files have similar annotations)
    title = input_data['annotations']['title']
    x_label = input_data['annotations']['x_label']
    y_label = input_data['annotations']['y_label']
    runNumber = unpack(newXpruned[0])[0]
    plt.xlim(min(newXpruned_new), max(newXpruned_new))
    if f_ymin is not None and f_ymax is None:
        plt.ylim(f_ymin, max(glob_y))
    elif f_ymin is None and f_ymax is not None:
        plt.ylim(min(glob_y), f_ymax)
    else:    
        plt.ylim(min(glob_y), max(glob_y))
    
    ax.set_xlabel(xlabel,fontsize = xlabel_font_size, loc=xlabel_location) #, labelpad=25

    ax.set_ylabel(ylabel,fontsize = ylabel_font_size, loc=ylabel_location)
    

    leg = ax.legend(facecolor='white',edgecolor="black",frameon=True, title=f"Run {runNumber}",fontsize=8) #,ncol=2 , title="Expected"
    leg.get_title().set_fontsize(8)  
    plt.savefig(output_file + ".pdf")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot data from multiple JSON files.")
    parser.add_argument('-o',"--outputName", type=str, help="name for the output plot image.", default="out")
    parser.add_argument("--db", type=str, help="db object", nargs='+', default="")
    parser.add_argument('-t',"--tags", type=str, help="tags for db object", nargs='+', default=None)
    parser.add_argument('-l',"--labels", type=str, help="labels for db object", nargs='+', default=None)
    parser.add_argument('-p',"--plugins", type=str, help="plugin", nargs='+', default=None)
    parser.add_argument('-tp',"--time_types", type=str, help="time_type", nargs='+', default=None)
    parser.add_argument('--unitTest', action="store_true", help="Enable if you want to do the unit test")
    parser.add_argument('--test', action="store_false", help="Enable if you want to do the test")
    parser.add_argument('--setRangeYMax', type=float, help='Enforce max for y-axis.', default=None)
    parser.add_argument('--setRangeYMin', type=float, help='Enforce min for y-axis.', default=None)
    args = parser.parse_args()

    input_files = args.db
    plugins = args.plugins
    labels = args.labels
    test = args.test
    outputName = args.outputName
    if args.unitTest:
        cmd = "conddb_import -c sqlite_file:BeamSpotObjects_FTV_GT_DigiMorphing_HLT_v0.db -f frontier://FrontierProd/CMS_CONDITIONS -i BeamSpotObjects_FTV_GT_DigiMorphing_HLT_v0 -b 1706957442384272"
        os.system(cmd)
        input_files = ["BeamSpotObjects_FTV_GT_DigiMorphing_HLT_v0.db"]
        outputName = "BeamSpotObjects_FTV_GT_DigiMorphing_HLT_v0"
    if not (args.tags is None):
        tags = args.tags
    else:
        tags = []
        for file in input_files:
            tags.append(file.split('.')[0])


    if not (args.time_types is None):
        time_types = args.time_types
    else:
        time_types = []
        for i in range(len(input_files)):
            time_types.append("Lumi")
    if not (args.plugins is None):
        plugins = args.plugins
    else:
        plugins = []
        for i in range(len(input_files)):
            plugins.append("pluginBeamSpot_PayloadInspector")
    if not (args.labels is None):
        labels = args.labels
    else:
        labels = []
        for file in input_files:
            label = file.split('.')[0].split('FTV_')[-1]
            labels.append(label)
    cmd = "eval $(scram ru -sh)"
    os.system(cmd)
    iovs = []
    for file in input_files:
        name = file.split('.')[0]
        cmd = f"conddb --db  {file} list {name}"
        output = subprocess.check_output([cmd], text=True, shell=True)
        lines = output.splitlines()
        count = 0
        for line in lines:
            if ("BeamSpotObjects" in line):
                if count == 0:
                    start_iov = line.split('(')[1].split(')')[0]
                    count += 1
                end_iov = line.split('(')[1].split(')')[0]
        iov = "{" + f'\"start_iov\": \"{start_iov}\", \"end_iov\": \"{end_iov}\"' + "}"
        iovs.append(iov)

    ylabels = ["X", "Y", "Z", "SigmaX", "SigmaY", "SigmaZ", "dXdZ",  "dYdZ"] 
    
    if ((len(input_files) == len(iovs)) and ((len(input_files) == len(tags))) and (len(input_files) == len(plugins)) and (len(input_files) == len(time_types))) and (len(input_files) == len(labels)):
        for ylabel in ylabels:
            for file, iv, tag, plugin, time_type in zip(input_files, iovs, tags, plugins, time_types):
                if (test):
                    
                    cmd = f"getPayloadData.py --plugin {plugin} --plot plot_BeamSpot_History{ylabel} --tag {tag} --time_type {time_type}  --iovs \'{iv}\' --db sqlite:{file} --test > toPlot_{ylabel}_{tag}.txt 2>&1"
                else:
                    cmd = f"getPayloadData.py --plugin {plugin} --plot plot_BeamSpot_History{ylabel} --tag {tag} --time_type {time_type}  --iovs \'{iv}\' --db sqlite:{file} > toPlot_{ylabel}_{tag}.txt 2>&1"
                os.system(cmd)
    else:
        raise ValueError(f"the number of input files: {len(input_files)}, tags: {len(tags)}, plugins: {len(plugins)}, time_types: {len(time_types)}, iovs: {len(iovs)}, labels: {len(labels)} are not the same")
    

    for ylabel in ylabels:
        files = []
        for tag in tags:
            files.append(f"toPlot_{ylabel}_{tag}.txt")
        plot(files, labels, f"plot{ylabel}_{outputName}", args.setRangeYMin, args.setRangeYMax)
