import pandas as pd
import json
import os
import glob

files = glob.glob(os.getcwd()+"/CondCore/CTPPSPlugins/test/results/*.json")       
print(os.getcwd())

for file in files:
    f = open(file)
    data = json.load(f)
    df = pd.DataFrame(data['data'])
    plot = df.plot.scatter(x='x', y='y', xlabel=data['annotations']['x_label'], ylabel=data['annotations']['y_label'])
    plot.get_figure().savefig(file.strip(".json")+".png")