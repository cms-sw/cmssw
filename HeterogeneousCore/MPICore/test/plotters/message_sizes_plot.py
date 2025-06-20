import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re

# Read the file manually line-by-line
timestamps = []
sizes = []
destinations = []

with open('mpi_message_sizes.txt', 'r') as f:
    for line in f:
        match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| Size: (\d+) bytes, Destination: (\d+)', line)
        if match:
            timestamps.append(match.group(1))
            sizes.append(int(match.group(2)))
            destinations.append(int(match.group(3)))

# Create a DataFrame
df = pd.DataFrame({
    'timestamp': pd.to_datetime(timestamps, format="%Y-%m-%d %H:%M:%S.%f"),
    'size': [size/1024 for size in sizes],
    'destination': destinations
})

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Scatter plot, one color per destination, semi-transparent
for dest, group in df.groupby('destination'):
    if dest:
        label = "from remote to local"
    else:
        label = "from local to remote"
    ax.scatter(group['timestamp'], group['size'], label=label, s=3, alpha=0.8)  # alpha=0.5 for half-transparent

ax.set_xlabel('Time')
ax.set_ylabel('Message Size (Kb)')
ax.set_title('MPI Message Sizes Over Time')
ax.legend()
ax.grid(True)

# Format x-axis nicely
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S.%f"))
fig.autofmt_xdate()

# Save the figure
plt.savefig('mpi_message_sizes_plot.png', dpi=300, bbox_inches='tight')
