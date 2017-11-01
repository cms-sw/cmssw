Sample: cannyEdgeDetectorNPP
Minimum spec: SM 2.0

An NPP CUDA Sample that demonstrates the recommended parameters to use with the nppiFilterCannyBorder_8u_C1R Canny Edge Detection image filter function. This function expects a single channel 8-bit grayscale input image. You can generate a grayscale image from a color image by first calling nppiColorToGray() or nppiRGBToGray(). The Canny Edge Detection function combines and improves on the techniques required to produce an edge detection image using multiple steps.

Key concepts:
Performance Strategies
Image Processing
NPP Library
