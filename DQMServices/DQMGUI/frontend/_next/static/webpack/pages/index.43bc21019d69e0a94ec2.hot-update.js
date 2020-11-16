webpackHotUpdate_N_E("pages/index",{

/***/ "./components/overlayWithAnotherPlot/index.tsx":
/*!*****************************************************!*\
  !*** ./components/overlayWithAnotherPlot/index.tsx ***!
  \*****************************************************/
/*! exports provided: OverlayWithAnotherPlot */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "OverlayWithAnotherPlot", function() { return OverlayWithAnotherPlot; });
/* harmony import */ var _babel_runtime_helpers_esm_toConsumableArray__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/toConsumableArray */ "./node_modules/@babel/runtime/helpers/esm/toConsumableArray.js");
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! antd/lib/modal/Modal */ "./node_modules/antd/lib/modal/Modal.js");
/* harmony import */ var antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_4___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_4__);
/* harmony import */ var _containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../containers/display/styledComponents */ "./containers/display/styledComponents.tsx");
/* harmony import */ var _containers_display_utils__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../containers/display/utils */ "./containers/display/utils.ts");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");
/* harmony import */ var _hooks_useRequest__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../../hooks/useRequest */ "./hooks/useRequest.tsx");
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _containers_display_content_folderPath__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! ../../containers/display/content/folderPath */ "./containers/display/content/folderPath.tsx");



var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/overlayWithAnotherPlot/index.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_2__["createElement"];









var OverlayWithAnotherPlot = function OverlayWithAnotherPlot(_ref) {
  _s();

  var visible = _ref.visible,
      setOpenOverlayWithAnotherPlotModal = _ref.setOpenOverlayWithAnotherPlotModal;

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_2__["useState"]([]),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState, 2),
      data = _React$useState2[0],
      setData = _React$useState2[1];

  var _React$useState3 = react__WEBPACK_IMPORTED_MODULE_2__["useState"]({
    folder_path: '',
    name: ''
  }),
      _React$useState4 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState3, 2),
      overlaidPlots = _React$useState4[0],
      setOverlaidPlots = _React$useState4[1];

  var _React$useState5 = react__WEBPACK_IMPORTED_MODULE_2__["useState"]([]),
      _React$useState6 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState5, 2),
      folderPath = _React$useState6[0],
      setFolderPath = _React$useState6[1];

  var _React$useState7 = react__WEBPACK_IMPORTED_MODULE_2__["useState"](''),
      _React$useState8 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState7, 2),
      currentFolder = _React$useState8[0],
      setCurrentFolder = _React$useState8[1];

  var _React$useState9 = react__WEBPACK_IMPORTED_MODULE_2__["useState"]({}),
      _React$useState10 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState9, 2),
      plot = _React$useState10[0],
      setPlot = _React$useState10[1];

  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_4__["useRouter"])();
  var query = router.query;

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_2__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_7__["store"]),
      updated_by_not_older_than = _React$useContext.updated_by_not_older_than;

  var params = {
    dataset_name: query.dataset_name,
    run_number: query.run_number,
    notOlderThan: updated_by_not_older_than,
    folders_path: overlaidPlots.folder_path,
    plot_name: overlaidPlots.name
  };
  var api = Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_6__["choose_api"])(params);
  var data_get_by_mount = Object(_hooks_useRequest__WEBPACK_IMPORTED_MODULE_8__["useRequest"])(api, {}, [overlaidPlots.folder_path]);
  react__WEBPACK_IMPORTED_MODULE_2__["useEffect"](function () {
    if (data_get_by_mount && data_get_by_mount.data) {
      setData(data_get_by_mount.data.data);
    }

    console.log();
  }, [data_get_by_mount.data]);
  react__WEBPACK_IMPORTED_MODULE_2__["useEffect"](function () {
    var copy = Object(_babel_runtime_helpers_esm_toConsumableArray__WEBPACK_IMPORTED_MODULE_0__["default"])(folderPath);

    var newItemIndexInFolderPath = copy.indexOf(currentFolder);
    console.log(newItemIndexInFolderPath, newItemIndexInFolderPath, copy);

    if (newItemIndexInFolderPath > -1) {
      var howManyItemsNeedToRemove = copy.length - 1 - newItemIndexInFolderPath;
      console.log(howManyItemsNeedToRemove);
      copy.splice(newItemIndexInFolderPath, howManyItemsNeedToRemove);
    } else {
      copy.push(currentFolder);
    }

    setFolderPath(copy);
    return function () {
      return setFolderPath([]);
    };
  }, [currentFolder, folderPath[folderPath.length - 1]]); // when the last folder in path is changed

  react__WEBPACK_IMPORTED_MODULE_2__["useEffect"](function () {
    var joinedFoldersForRequest = folderPath.join('/').substr(1);
    console.log(joinedFoldersForRequest);
    setOverlaidPlots({
      name: '',
      folder_path: joinedFoldersForRequest
    });
  }, folderPath);

  var changeFolderPathByBreadcrumb = function changeFolderPathByBreadcrumb(parameters) {
    console.log(parameters);

    if (parameters.folder_path === '/') {
      setFolderPath([]);
      setCurrentFolder('');
    }

    var folders = overlaidPlots.folder_path.split('/');
    setFolderPath(folders);
  };

  return __jsx(antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_3___default.a, {
    visible: visible,
    onCancel: function onCancel() {
      setOpenOverlayWithAnotherPlotModal(false);
      setFolderPath([]);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 86,
      columnNumber: 5
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Row"], {
    gutter: 16,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 93,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Col"], {
    style: {
      padding: 8
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 94,
      columnNumber: 9
    }
  }, __jsx(_containers_display_content_folderPath__WEBPACK_IMPORTED_MODULE_10__["FolderPath"], {
    folder_path: overlaidPlots.folder_path,
    changeFolderPathByBreadcrumb: changeFolderPathByBreadcrumb,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 95,
      columnNumber: 11
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Row"], {
    style: {
      width: '100%'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 97,
      columnNumber: 9
    }
  }, data.map(function (folder_or_plot) {
    return __jsx(react__WEBPACK_IMPORTED_MODULE_2__["Fragment"], null, folder_or_plot.subdir && __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Col"], {
      span: 8,
      onClick: function onClick() {
        return setCurrentFolder(folder_or_plot.subdir);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 103,
        columnNumber: 21
      }
    }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["Icon"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 104,
        columnNumber: 23
      }
    }), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_5__["StyledA"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 105,
        columnNumber: 23
      }
    }, folder_or_plot.subdir)));
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Row"], {
    style: {
      width: '100%'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 113,
      columnNumber: 9
    }
  }, data.map(function (folder_or_plot) {
    return __jsx(react__WEBPACK_IMPORTED_MODULE_2__["Fragment"], null, folder_or_plot.name && __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Col"], {
      span: 8,
      onClick: function onClick() {
        return setPlot(folder_or_plot);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 119,
        columnNumber: 21
      }
    }, __jsx(antd__WEBPACK_IMPORTED_MODULE_9__["Button"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 120,
        columnNumber: 23
      }
    }, folder_or_plot.name)));
  }))));
};

_s(OverlayWithAnotherPlot, "2chABmc/YJS6AMKuh3dCNIbNV40=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_4__["useRouter"], _hooks_useRequest__WEBPACK_IMPORTED_MODULE_8__["useRequest"]];
});

_c = OverlayWithAnotherPlot;

var _c;

$RefreshReg$(_c, "OverlayWithAnotherPlot");

;
    var _a, _b;
    // Legacy CSS implementations will `eval` browser code in a Node.js context
    // to extract CSS. For backwards compatibility, we need to check we're in a
    // browser context before continuing.
    if (typeof self !== 'undefined' &&
        // AMP / No-JS mode does not inject these helpers:
        '$RefreshHelpers$' in self) {
        var currentExports = module.__proto__.exports;
        var prevExports = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevExports) !== null && _b !== void 0 ? _b : null;
        // This cannot happen in MainTemplate because the exports mismatch between
        // templating and execution.
        self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.i);
        // A module can be accepted automatically based on its exports, e.g. when
        // it is a Refresh Boundary.
        if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {
            // Save the previous exports on update so we can compare the boundary
            // signatures.
            module.hot.dispose(function (data) {
                data.prevExports = currentExports;
            });
            // Unconditionally accept an update to this module, we'll check if it's
            // still a Refresh Boundary later.
            module.hot.accept();
            // This field is set when the previous version of this module was a
            // Refresh Boundary, letting us know we need to check for invalidation or
            // enqueue an update.
            if (prevExports !== null) {
                // A boundary can become ineligible if its exports are incompatible
                // with the previous exports.
                //
                // For example, if you add/remove/change exports, we'll want to
                // re-execute the importing modules, and force those components to
                // re-render. Similarly, if you convert a class component to a
                // function, we want to invalidate the boundary.
                if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevExports, currentExports)) {
                    module.hot.invalidate();
                }
                else {
                    self.$RefreshHelpers$.scheduleUpdate();
                }
            }
        }
        else {
            // Since we just executed the code for the module, it's possible that the
            // new exports made it ineligible for being a boundary.
            // We only care about the case when we were _previously_ a boundary,
            // because we already accepted this update (accidental side effect).
            var isNoLongerABoundary = prevExports !== null;
            if (isNoLongerABoundary) {
                module.hot.invalidate();
            }
        }
    }

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9vdmVybGF5V2l0aEFub3RoZXJQbG90L2luZGV4LnRzeCJdLCJuYW1lcyI6WyJPdmVybGF5V2l0aEFub3RoZXJQbG90IiwidmlzaWJsZSIsInNldE9wZW5PdmVybGF5V2l0aEFub3RoZXJQbG90TW9kYWwiLCJSZWFjdCIsImRhdGEiLCJzZXREYXRhIiwiZm9sZGVyX3BhdGgiLCJuYW1lIiwib3ZlcmxhaWRQbG90cyIsInNldE92ZXJsYWlkUGxvdHMiLCJmb2xkZXJQYXRoIiwic2V0Rm9sZGVyUGF0aCIsImN1cnJlbnRGb2xkZXIiLCJzZXRDdXJyZW50Rm9sZGVyIiwicGxvdCIsInNldFBsb3QiLCJyb3V0ZXIiLCJ1c2VSb3V0ZXIiLCJxdWVyeSIsInN0b3JlIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsInBhcmFtcyIsImRhdGFzZXRfbmFtZSIsInJ1bl9udW1iZXIiLCJub3RPbGRlclRoYW4iLCJmb2xkZXJzX3BhdGgiLCJwbG90X25hbWUiLCJhcGkiLCJjaG9vc2VfYXBpIiwiZGF0YV9nZXRfYnlfbW91bnQiLCJ1c2VSZXF1ZXN0IiwiY29uc29sZSIsImxvZyIsImNvcHkiLCJuZXdJdGVtSW5kZXhJbkZvbGRlclBhdGgiLCJpbmRleE9mIiwiaG93TWFueUl0ZW1zTmVlZFRvUmVtb3ZlIiwibGVuZ3RoIiwic3BsaWNlIiwicHVzaCIsImpvaW5lZEZvbGRlcnNGb3JSZXF1ZXN0Iiwiam9pbiIsInN1YnN0ciIsImNoYW5nZUZvbGRlclBhdGhCeUJyZWFkY3J1bWIiLCJwYXJhbWV0ZXJzIiwiZm9sZGVycyIsInNwbGl0IiwicGFkZGluZyIsIndpZHRoIiwibWFwIiwiZm9sZGVyX29yX3Bsb3QiLCJzdWJkaXIiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBQ0E7QUFHQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFRTyxJQUFNQSxzQkFB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O0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FDRSxNQUFDLDJDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FBVUEsY0FBYyxDQUFDM0MsSUFBekIsQ0FERixDQUZKLENBREY7QUFTRCxHQVZELENBRkosQ0FwQkYsQ0FQRixDQURGO0FBOENELENBaEhNOztHQUFNUCxzQjtVQU9JaUIscUQsRUFhV2EsNEQ7OztLQXBCZjlCLHNCIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjQzYmMyMTAxOWQ2OWUwYTk0ZWMyLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCdcclxuaW1wb3J0IE1vZGFsIGZyb20gJ2FudGQvbGliL21vZGFsL01vZGFsJ1xyXG5pbXBvcnQgeyB1c2VSb3V0ZXIgfSBmcm9tICduZXh0L3JvdXRlcidcclxuXHJcbmltcG9ydCB7IFBhcmFtc0ZvckFwaVByb3BzLCBQbG90b3ZlcmxhaWRTZXBhcmF0ZWx5UHJvcHMsIFF1ZXJ5UHJvcHMgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcydcclxuaW1wb3J0IHsgSWNvbiwgU3R5bGVkQSB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9zdHlsZWRDb21wb25lbnRzJ1xyXG5pbXBvcnQgeyBjaG9vc2VfYXBpIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L3V0aWxzJ1xyXG5pbXBvcnQgeyBzdG9yZSB9IGZyb20gJy4uLy4uL2NvbnRleHRzL2xlZnRTaWRlQ29udGV4dCdcclxuaW1wb3J0IHsgdXNlUmVxdWVzdCB9IGZyb20gJy4uLy4uL2hvb2tzL3VzZVJlcXVlc3QnXHJcbmltcG9ydCB7IEJ1dHRvbiwgQ29sLCBSb3cgfSBmcm9tICdhbnRkJ1xyXG5pbXBvcnQgeyBGb2xkZXJQYXRoIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2NvbnRlbnQvZm9sZGVyUGF0aCdcclxuaW1wb3J0IHsgUGFyc2VkVXJsUXVlcnlJbnB1dCB9IGZyb20gJ3F1ZXJ5c3RyaW5nJ1xyXG5cclxuaW50ZXJmYWNlIE92ZXJsYXlXaXRoQW5vdGhlclBsb3RQcm9wcyB7XHJcbiAgdmlzaWJsZTogYm9vbGVhbjtcclxuICBzZXRPcGVuT3ZlcmxheVdpdGhBbm90aGVyUGxvdE1vZGFsOiBhbnlcclxufVxyXG5cclxuZXhwb3J0IGNvbnN0IE92ZXJsYXlXaXRoQW5vdGhlclBsb3QgPSAoeyB2aXNpYmxlLCBzZXRPcGVuT3ZlcmxheVdpdGhBbm90aGVyUGxvdE1vZGFsIH06IE92ZXJsYXlXaXRoQW5vdGhlclBsb3RQcm9wcykgPT4ge1xyXG4gIGNvbnN0IFtkYXRhLCBzZXREYXRhXSA9IFJlYWN0LnVzZVN0YXRlPGFueT4oW10pXHJcbiAgY29uc3QgW292ZXJsYWlkUGxvdHMsIHNldE92ZXJsYWlkUGxvdHNdID0gUmVhY3QudXNlU3RhdGU8UGxvdG92ZXJsYWlkU2VwYXJhdGVseVByb3BzPih7IGZvbGRlcl9wYXRoOiAnJywgbmFtZTogJycgfSlcclxuICBjb25zdCBbZm9sZGVyUGF0aCwgc2V0Rm9sZGVyUGF0aF0gPSBSZWFjdC51c2VTdGF0ZTxzdHJpbmdbXT4oW10pXHJcbiAgY29uc3QgW2N1cnJlbnRGb2xkZXIsIHNldEN1cnJlbnRGb2xkZXJdID0gUmVhY3QudXNlU3RhdGUoJycpXHJcbiAgY29uc3QgW3Bsb3QsIHNldFBsb3RdID0gUmVhY3QudXNlU3RhdGUoe30pXHJcblxyXG4gIGNvbnN0IHJvdXRlciA9IHVzZVJvdXRlcigpO1xyXG4gIGNvbnN0IHF1ZXJ5OiBRdWVyeVByb3BzID0gcm91dGVyLnF1ZXJ5O1xyXG4gIGNvbnN0IHsgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiB9ID0gUmVhY3QudXNlQ29udGV4dChzdG9yZSlcclxuXHJcbiAgY29uc3QgcGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wcyA9IHtcclxuICAgIGRhdGFzZXRfbmFtZTogcXVlcnkuZGF0YXNldF9uYW1lIGFzIHN0cmluZyxcclxuICAgIHJ1bl9udW1iZXI6IHF1ZXJ5LnJ1bl9udW1iZXIgYXMgc3RyaW5nLFxyXG4gICAgbm90T2xkZXJUaGFuOiB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuLFxyXG4gICAgZm9sZGVyc19wYXRoOiBvdmVybGFpZFBsb3RzLmZvbGRlcl9wYXRoLFxyXG4gICAgcGxvdF9uYW1lOiBvdmVybGFpZFBsb3RzLm5hbWVcclxuICB9XHJcblxyXG4gIGNvbnN0IGFwaSA9IGNob29zZV9hcGkocGFyYW1zKVxyXG4gIGNvbnN0IGRhdGFfZ2V0X2J5X21vdW50ID0gdXNlUmVxdWVzdChhcGksXHJcbiAgICB7fSxcclxuICAgIFtvdmVybGFpZFBsb3RzLmZvbGRlcl9wYXRoXVxyXG4gICk7XHJcblxyXG4gIFJlYWN0LnVzZUVmZmVjdCgoKSA9PiB7XHJcbiAgICBpZiAoZGF0YV9nZXRfYnlfbW91bnQgJiYgZGF0YV9nZXRfYnlfbW91bnQuZGF0YSkge1xyXG4gICAgICBzZXREYXRhKGRhdGFfZ2V0X2J5X21vdW50LmRhdGEuZGF0YSlcclxuICAgIH1cclxuICAgIGNvbnNvbGUubG9nKClcclxuICB9LCBbZGF0YV9nZXRfYnlfbW91bnQuZGF0YV0pXHJcblxyXG4gIFJlYWN0LnVzZUVmZmVjdCgoKSA9PiB7XHJcbiAgICBjb25zdCBjb3B5ID0gWy4uLmZvbGRlclBhdGhdXHJcbiAgICBjb25zdCBuZXdJdGVtSW5kZXhJbkZvbGRlclBhdGggPSBjb3B5LmluZGV4T2YoY3VycmVudEZvbGRlcilcclxuICAgIGNvbnNvbGUubG9nKG5ld0l0ZW1JbmRleEluRm9sZGVyUGF0aCwgbmV3SXRlbUluZGV4SW5Gb2xkZXJQYXRoLCBjb3B5KVxyXG5cclxuICAgIGlmIChuZXdJdGVtSW5kZXhJbkZvbGRlclBhdGggPiAtMSkge1xyXG4gICAgICBjb25zdCBob3dNYW55SXRlbXNOZWVkVG9SZW1vdmUgPSAoY29weS5sZW5ndGggLSAxKSAtIG5ld0l0ZW1JbmRleEluRm9sZGVyUGF0aFxyXG4gICAgICBjb25zb2xlLmxvZyhob3dNYW55SXRlbXNOZWVkVG9SZW1vdmUpXHJcbiAgICAgIGNvcHkuc3BsaWNlKG5ld0l0ZW1JbmRleEluRm9sZGVyUGF0aCwgaG93TWFueUl0ZW1zTmVlZFRvUmVtb3ZlKVxyXG4gICAgfVxyXG4gICAgZWxzZSB7XHJcbiAgICAgIGNvcHkucHVzaChjdXJyZW50Rm9sZGVyKVxyXG4gICAgfVxyXG4gICAgc2V0Rm9sZGVyUGF0aChjb3B5KVxyXG4gICAgcmV0dXJuICgpID0+IHNldEZvbGRlclBhdGgoW10pXHJcbiAgfSwgW2N1cnJlbnRGb2xkZXIsIGZvbGRlclBhdGhbZm9sZGVyUGF0aC5sZW5ndGggLSAxXV0pIC8vIHdoZW4gdGhlIGxhc3QgZm9sZGVyIGluIHBhdGggaXMgY2hhbmdlZFxyXG5cclxuICBSZWFjdC51c2VFZmZlY3QoKCkgPT4ge1xyXG4gICAgY29uc3Qgam9pbmVkRm9sZGVyc0ZvclJlcXVlc3QgPSBmb2xkZXJQYXRoLmpvaW4oJy8nKS5zdWJzdHIoMSlcclxuICAgIGNvbnNvbGUubG9nKGpvaW5lZEZvbGRlcnNGb3JSZXF1ZXN0KVxyXG4gICAgc2V0T3ZlcmxhaWRQbG90cyh7IG5hbWU6ICcnLCBmb2xkZXJfcGF0aDogam9pbmVkRm9sZGVyc0ZvclJlcXVlc3QgfSlcclxuICB9LCBmb2xkZXJQYXRoKVxyXG5cclxuICBjb25zdCBjaGFuZ2VGb2xkZXJQYXRoQnlCcmVhZGNydW1iID0gKHBhcmFtZXRlcnM6IFBhcnNlZFVybFF1ZXJ5SW5wdXQpID0+IHtcclxuICAgIGNvbnNvbGUubG9nKHBhcmFtZXRlcnMpXHJcbiAgICBpZiAocGFyYW1ldGVycy5mb2xkZXJfcGF0aCA9PT0gJy8nKSB7XHJcbiAgICAgIHNldEZvbGRlclBhdGgoW10pXHJcbiAgICAgIHNldEN1cnJlbnRGb2xkZXIoJycpXHJcbiAgICB9XHJcbiAgICBjb25zdCBmb2xkZXJzID0gb3ZlcmxhaWRQbG90cy5mb2xkZXJfcGF0aC5zcGxpdCgnLycpXHJcbiAgICBzZXRGb2xkZXJQYXRoKGZvbGRlcnMpXHJcbiAgfVxyXG5cclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxNb2RhbFxyXG4gICAgICB2aXNpYmxlPXt2aXNpYmxlfVxyXG4gICAgICBvbkNhbmNlbD17KCkgPT4ge1xyXG4gICAgICAgIHNldE9wZW5PdmVybGF5V2l0aEFub3RoZXJQbG90TW9kYWwoZmFsc2UpXHJcbiAgICAgICAgc2V0Rm9sZGVyUGF0aChbXSlcclxuICAgICAgfX1cclxuICAgID5cclxuICAgICAgPFJvdyBndXR0ZXI9ezE2fT5cclxuICAgICAgICA8Q29sIHN0eWxlPXt7IHBhZGRpbmc6IDggfX0+XHJcbiAgICAgICAgICA8Rm9sZGVyUGF0aCBmb2xkZXJfcGF0aD17b3ZlcmxhaWRQbG90cy5mb2xkZXJfcGF0aH0gY2hhbmdlRm9sZGVyUGF0aEJ5QnJlYWRjcnVtYj17Y2hhbmdlRm9sZGVyUGF0aEJ5QnJlYWRjcnVtYn0gLz5cclxuICAgICAgICA8L0NvbD5cclxuICAgICAgICA8Um93IHN0eWxlPXt7IHdpZHRoOiAnMTAwJScgfX0+XHJcbiAgICAgICAgICB7XHJcbiAgICAgICAgICAgIGRhdGEubWFwKChmb2xkZXJfb3JfcGxvdDogYW55KSA9PiB7XHJcbiAgICAgICAgICAgICAgcmV0dXJuIChcclxuICAgICAgICAgICAgICAgIDw+XHJcbiAgICAgICAgICAgICAgICAgIHtmb2xkZXJfb3JfcGxvdC5zdWJkaXIgJiZcclxuICAgICAgICAgICAgICAgICAgICA8Q29sIHNwYW49ezh9IG9uQ2xpY2s9eygpID0+IHNldEN1cnJlbnRGb2xkZXIoZm9sZGVyX29yX3Bsb3Quc3ViZGlyKX0+XHJcbiAgICAgICAgICAgICAgICAgICAgICA8SWNvbiAvPlxyXG4gICAgICAgICAgICAgICAgICAgICAgPFN0eWxlZEE+e2ZvbGRlcl9vcl9wbG90LnN1YmRpcn08L1N0eWxlZEE+XHJcbiAgICAgICAgICAgICAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIDwvPlxyXG4gICAgICAgICAgICAgIClcclxuICAgICAgICAgICAgfSlcclxuICAgICAgICAgIH1cclxuICAgICAgICA8L1Jvdz5cclxuICAgICAgICA8Um93IHN0eWxlPXt7IHdpZHRoOiAnMTAwJScgfX0+XHJcbiAgICAgICAgICB7XHJcbiAgICAgICAgICAgIGRhdGEubWFwKChmb2xkZXJfb3JfcGxvdDogYW55KSA9PiB7XHJcbiAgICAgICAgICAgICAgcmV0dXJuIChcclxuICAgICAgICAgICAgICAgIDw+XHJcbiAgICAgICAgICAgICAgICAgIHtmb2xkZXJfb3JfcGxvdC5uYW1lICYmXHJcbiAgICAgICAgICAgICAgICAgICAgPENvbCBzcGFuPXs4fSBvbkNsaWNrPXsoKSA9PiBzZXRQbG90KGZvbGRlcl9vcl9wbG90KX0+XHJcbiAgICAgICAgICAgICAgICAgICAgICA8QnV0dG9uID57Zm9sZGVyX29yX3Bsb3QubmFtZX08L0J1dHRvbj5cclxuICAgICAgICAgICAgICAgICAgICA8L0NvbD5cclxuICAgICAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgICAgPC8+XHJcbiAgICAgICAgICAgICAgKVxyXG4gICAgICAgICAgICB9KVxyXG4gICAgICAgICAgfVxyXG4gICAgICAgIDwvUm93PlxyXG4gICAgICA8L1Jvdz5cclxuICAgIDwvTW9kYWw+XHJcbiAgKVxyXG59Il0sInNvdXJjZVJvb3QiOiIifQ==